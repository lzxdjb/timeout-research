"""Projector compatibility with the installed mbridge and Megatron checkpoints."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_module(name, relative_path):
    path = Path(__file__).resolve().parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audio = _load_module("projector_audio", "verl/models/mcore/frozen_audio.py")
mapping = _load_module("projector_bridge", "verl/models/mcore/audio_projector_bridge.py")


def _bridge(tp_size=2):
    from mbridge.models.qwen3_5.qwen3_5_vl_bridge import Qwen3_5MoeVlBridge

    # Use the actual conversion/export methods, without allocating the full LLM.
    bridge = object.__new__(Qwen3_5MoeVlBridge)
    bridge.config = SimpleNamespace(attention_output_gate=False, num_moe_experts=8)
    bridge.dtype = torch.float32
    bridge.make_vocab_size_divisible_by = None
    bridge.export_weights_buff = {}
    bridge.export_weights_buffer_max_size_bytes = 1024
    bridge.mpu = SimpleNamespace(
        tp_size=tp_size,
        tp_rank=0,
        tp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        pp_size=1,
        pp_rank=0,
    )
    return bridge


def _projector():
    torch.manual_seed(27)
    reference = torch.nn.Module()
    reference.proj1 = torch.nn.Linear(8, 8)
    reference.proj2 = torch.nn.Linear(8, 4)
    return audio.AudioProjector({"d_model": 8, "output_dim": 4, "activation_function": "gelu"}, reference.state_dict())


def test_legacy_mapping_is_instance_local_and_round_trips_all_projector_tensors():
    bridge, unrelated = _bridge(), _bridge()
    original_mapping = dict(unrelated._DIRECT_MAPPING)
    mapping.configure_audio_projector_bridge(bridge)
    assert unrelated._DIRECT_MAPPING == original_mapping
    assert bridge._DIRECT_MAPPING is not unrelated._DIRECT_MAPPING
    for name, parameter in _projector().named_parameters():
        internal = f"_verl_audio_projector.{name}"
        external = f"model.audio_tower.{name}"
        assert bridge._weight_name_mapping_mcore_to_hf(internal) == [external]
        imported = bridge._weight_to_mcore_format(internal, [parameter.detach().clone()])
        torch.testing.assert_close(imported, parameter, atol=0, rtol=0)
        replicated = bridge._weight_split_across_tp(internal, imported, parameter, tp_split_size=2)
        assert len(replicated) == 2
        for replica in replicated:
            torch.testing.assert_close(replica, parameter, atol=0, rtol=0)
        names, tensors = bridge._weight_to_hf_format(internal, imported)
        assert names == [external]
        torch.testing.assert_close(tensors[0], parameter, atol=0, rtol=0)


def test_real_legacy_export_bypasses_tp_merge_for_replicated_projector(monkeypatch):
    from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes

    bridge = _bridge(tp_size=2)
    mapping.configure_audio_projector_bridge(bridge)
    model = torch.nn.Module()
    model._verl_audio_projector = _projector()
    for parameter in model.parameters():
        set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
        assert parameter.tensor_model_parallel is False

    def forbidden_merge(*_args, **_kwargs):
        pytest.fail("Replicated projectors must bypass the TP-sharded merge path")

    monkeypatch.setattr(bridge, "_weight_merge_across_tp", forbidden_merge)
    exported = dict(bridge.export_weights([model]))
    expected = {
        f"model.audio_tower.{name}": tensor for name, tensor in model._verl_audio_projector.state_dict().items()
    }
    assert exported.keys() == expected.keys()
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
        assert not tensor.requires_grad
    # Fast HF saving identifies these as ordinary replicated parameters too.
    without_gather = list(bridge.export_weights_without_gather([model]))
    assert len(without_gather) == 4
    assert all(item[1:5] == (0, 0, 0, 0) and item[5] is False for item in without_gather)


def test_real_megatron_sharded_state_includes_projector_and_optimizer_moments():
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        optim_state_to_sharding_state,
    )
    from megatron.core.transformer.module import MegatronModule

    model = MegatronModule(SimpleNamespace())
    # Explicit groups keep native checkpoint construction independent from a
    # running distributed job; no distributed collective is needed for metadata.
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    model.tp_group = group
    model._verl_audio_projector = _projector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model._verl_audio_projector(torch.randn(5, 8)).square().mean().backward()
    optimizer.step()
    sharded = model.sharded_state_dict(metadata={"dp_cp_group": group})
    parameters = dict(model.named_parameters())
    assert sharded.keys() == parameters.keys()
    assert len(sharded) == 4
    for name, tensor in sharded.items():
        assert isinstance(tensor, ShardedTensor)
        assert tensor.data is parameters[name]
        assert tensor.global_shape == tuple(parameters[name].shape)
        assert tensor.global_offset == (0,) * parameters[name].ndim
        assert tensor.replica_id == (0, 0, 0)
    # Megatron's optimizer checkpoint conversion must retain the identity link
    # from these replicated parameters to their first/second Adam moments.
    param_map = get_param_id_to_sharded_param_map(sharded, model.parameters())
    assert len(param_map) == 4
    state = optimizer.state_dict()
    optim_state_to_sharding_state(state, param_map, exclude_keys=("step",))
    for parameter_state in state["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            assert isinstance(parameter_state[key], ShardedTensor)
            assert "_verl_audio_projector.proj" in parameter_state[key].key


def test_bridge_without_direct_mappings_fails_before_model_mutation():
    bridge = SimpleNamespace()
    with pytest.raises(ValueError, match="legacy mbridge direct parameter mappings"):
        mapping.configure_audio_projector_bridge(bridge)
    assert vars(bridge) == {}
