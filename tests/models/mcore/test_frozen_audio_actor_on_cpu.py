"""Numerical audio-conditioning regressions without a GPU or full student."""

import ast
import importlib.util
import json
import logging
from importlib.metadata import distribution
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional

import pytest
import torch
from safetensors.torch import save_file

_SPEC = importlib.util.spec_from_file_location(
    "frozen_actor_audio", Path(__file__).resolve().parents[3] / "verl/models/mcore/frozen_audio.py"
)
audio = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audio)


@pytest.fixture
def checkpoint(tmp_path):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    torch.manual_seed(91)
    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=8,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        encoder_layers=1,
        num_hidden_layers=1,
        num_mel_bins=8,
        downsample_hidden_size=2,
        output_dim=4,
        max_source_positions=20,
        n_window=50,
        n_window_infer=100,
        initializer_range=0.2,
    )
    config._attn_implementation = "eager"
    tower = Qwen3OmniMoeAudioEncoder(config).eval()
    state = {"model.audio_tower." + name: tensor for name, tensor in tower.state_dict().items()}
    save_file(state, tmp_path / "audio.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "audio.safetensors" for name in state}})
    )
    hf = SimpleNamespace(audio_config=config.to_dict(), audio_token_id=19, text_config=SimpleNamespace(hidden_size=4))
    return tmp_path, hf, tower


class Embedding(torch.nn.Module):
    scatter_to_sequence_parallel = False
    reduce_scatter_embeddings = False

    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(20, 4)

    def forward(self, input_ids, position_ids=None):
        return self.word_embeddings(input_ids).transpose(0, 1).contiguous()


class Bridge(torch.nn.Module):
    pre_process = True

    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.embedding = Embedding()
        self.head = torch.nn.Linear(4, 1, bias=False)

    def forward(self, input_ids):
        # Mirrors the actual bridge boundary: first form full SBH embeddings,
        # then pack and scatter outside the embedding (tested separately below).
        values = self.language_model.embedding(input_ids=input_ids, position_ids=None)
        return self.head(values).sum()


def _features(lengths=(9,)):
    torch.manual_seed(19)
    features = torch.randn(len(lengths), 8, max(lengths))
    mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    return {"input_features": features, "feature_attention_mask": mask.long()}


def _installed(checkpoint):
    path, hf, _ = checkpoint
    bridge = Bridge()
    adapters = audio.install_frozen_audio_adapters([bridge], path, hf)
    return bridge, adapters[0]


@pytest.mark.parametrize("wrapped", [False, True])
def test_installed_mbridge_shadow_wrapper_uses_registered_embedding(checkpoint, wrapped):
    source = distribution("mbridge").locate_file("mbridge/models/qwen3_5/model.py")
    tree = ast.parse(source.read_text())
    shadow = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_SPScatterEmbeddingWrapper"
    )
    vl_model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5VLModel")
    forward = next(node for node in vl_model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    # Execute the installed bridge's exact embedding statement, rather than
    # mimicking public attribute access that the installed VL forward bypasses.
    statement = next(
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "combined_embeddings"
        and any(
            isinstance(child, ast.Subscript)
            and ast.unparse(child.value) == "self.language_model._modules"
            and isinstance(child.slice, ast.Constant)
            and child.slice.value == "embedding"
            for child in ast.walk(node.value)
        )
    )
    embedding_forward = ast.parse("def installed_embedding_forward(self, input_ids):\n    return None").body[0]
    embedding_forward.body = [statement, ast.Return(value=ast.Name(id="combined_embeddings", ctx=ast.Load()))]

    def forbidden_mtp_scatter(_output):
        raise AssertionError("The installed VL forward must bypass the MTP scatter wrapper")

    namespace = {"tensor_parallel": SimpleNamespace(scatter_to_sequence_parallel_region=forbidden_mtp_scatter)}
    module = ast.fix_missing_locations(ast.Module(body=[shadow, embedding_forward], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    bridge = Bridge()
    embedding = bridge.language_model._modules["embedding"]
    object.__setattr__(bridge.language_model, "embedding", namespace[shadow.name](embedding, do_scatter=True))
    assert not isinstance(bridge.language_model.embedding, torch.nn.Module)
    bridge.forward = MethodType(namespace[embedding_forward.name], bridge)

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, input_ids):
            return self.module(input_ids)

    model = Wrapper(Wrapper(bridge)) if wrapped else bridge
    names_before = tuple(model.state_dict())
    path, config, _ = checkpoint
    adapter = audio.install_frozen_audio_adapters([model], path, config)[0]
    ids = torch.tensor([[1, 19, 19, 2]])
    baseline = model(ids).detach()
    features = _features()
    encoded = adapter.encode(features["input_features"], features["feature_attention_mask"], ids.device)
    with audio.frozen_audio_forward(model, ids, features):
        conditioned = model(ids)
    torch.testing.assert_close(conditioned.transpose(0, 1)[ids == 19], encoded)
    assert not torch.equal(conditioned, baseline)
    bridge.head(conditioned).sum().backward()
    assert bridge.head.weight.grad.abs().sum() > 0
    assert embedding.word_embeddings.weight.grad[19].abs().sum() == 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in adapter.tower.parameters())
    assert tuple(model.state_dict()) == names_before
    assert not embedding._forward_hooks
    with audio.frozen_audio_forward(model, torch.tensor([[1, 2]]), {}):
        text = model(torch.tensor([[1, 2]]))
    torch.testing.assert_close(text, model(torch.tensor([[1, 2]])), atol=0, rtol=0)
    embedding.scatter_to_sequence_parallel = True
    with pytest.raises(ValueError, match="before embedding sequence-parallel"):
        audio.install_frozen_audio_adapters([model], path, config)


def test_real_tiny_hf_encoder_checkpoint_load_and_ragged_output(checkpoint):
    path, hf, reference = checkpoint
    adapter = audio.FrozenAudioAdapter(path, hf.audio_config, 19, 4)
    mm = _features((30, 117))
    features, mask = mm["input_features"], mm["feature_attention_mask"].bool()
    packed = features.transpose(1, 2)[mask].transpose(0, 1).contiguous()
    with torch.no_grad():
        expected = reference(packed, feature_lens=mask.sum(-1)).last_hidden_state
    actual = adapter.encode(features, mask, torch.device("cpu"))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (20, 4)
    assert not actual.requires_grad
    assert all(not parameter.requires_grad for parameter in adapter.tower.parameters())
    assert not any(parameter.is_meta for parameter in adapter.tower.parameters())
    assert not any(buffer.is_meta for buffer in adapter.tower.buffers())


def test_audio_changes_actor_and_language_gradients_but_remains_frozen(checkpoint):
    bridge, adapter = _installed(checkpoint)
    names_before = tuple(bridge.state_dict())
    ids = torch.tensor([[1, 19, 19, 2]])  # Nine mel frames produce two audio tokens.
    mm = _features()
    with audio.frozen_audio_forward(bridge, ids, mm):
        first = bridge(ids)
    first.backward()
    assert bridge.head.weight.grad is not None and bridge.head.weight.grad.abs().sum() > 0
    assert bridge.language_model.embedding.word_embeddings.weight.grad[19].abs().sum() == 0
    assert all(parameter.grad is None for parameter in adapter.tower.parameters())
    changed = {**mm, "input_features": mm["input_features"] * -3 + 1}
    with audio.frozen_audio_forward(bridge, ids, changed):
        second = bridge(ids)
    assert not torch.equal(first, second)
    assert tuple(bridge.state_dict()) == names_before  # No frozen weights leak into Megatron export.
    assert len(bridge.language_model.embedding._forward_hooks) == 0
    adapter.offload()
    assert next(adapter.tower.parameters()).device.type == "cpu"


def test_text_path_unchanged_and_does_not_load_audio_weights(checkpoint):
    bridge, adapter = _installed(checkpoint)
    ids = torch.tensor([[1, 2, 3]])
    reference = bridge(ids)
    with audio.frozen_audio_forward(bridge, ids, {}):
        actual = bridge(ids)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert adapter.tower is None


def test_first_audio_diagnostics_remain_visible_with_global_warn(checkpoint, caplog):
    bridge, adapter = _installed(checkpoint)
    ids = torch.tensor([[1, 19, 19, 2]])
    assert audio.logger.level == logging.INFO
    with caplog.at_level(logging.INFO):
        logging.getLogger().setLevel(logging.WARNING)
        for _ in range(2):
            with audio.frozen_audio_forward(bridge, ids, _features()):
                bridge(ids)
    messages = [record.message for record in caplog.records if record.name == audio.logger.name]
    assert len(messages) == 2
    assert f"tensors={len(adapter.files)}" in messages[0]
    assert "parameters=" in messages[0]
    assert "clips=1 features_shape=(1, 8, 9) mask_shape=(1, 9) embeddings=2 hidden=4" in messages[1]
    assert all("encoder_grad=False projector_grad=False" in message for message in messages)


def test_audio_clip_alignment_in_a_mixed_text_audio_batch(checkpoint):
    bridge, adapter = _installed(checkpoint)
    ids = torch.tensor([[1, 2, 3, 0, 0], [1, 19, 19, 2, 0], [1, 19, 19, 19, 2]])
    mm = _features((9, 17))
    original = bridge.language_model.embedding(input_ids=ids).transpose(0, 1).detach()
    encoded = adapter.encode(mm["input_features"], mm["feature_attention_mask"], ids.device)
    with audio.frozen_audio_forward(bridge, ids, mm):
        result = bridge.language_model.embedding(input_ids=ids).transpose(0, 1)
    torch.testing.assert_close(result[ids == 19], encoded)
    torch.testing.assert_close(result[ids != 19], original[ids != 19])
    mismatched = ids.clone()
    mismatched[1] = torch.tensor([1, 19, 19, 19, 2])
    mismatched[2] = torch.tensor([1, 19, 19, 2, 0])
    with pytest.raises(ValueError, match="per-clip feature lengths"):
        with audio.frozen_audio_forward(bridge, mismatched, mm):
            pass


def test_audio_placeholders_cannot_silently_lose_features(checkpoint):
    bridge, _ = _installed(checkpoint)
    with pytest.raises(ValueError, match="refusing unconditioned"):
        with audio.frozen_audio_forward(bridge, torch.tensor([[19, 19]]), {}):
            pass


def test_disabled_audio_path_preserves_existing_forwarding():
    bridge = Bridge()
    ids = torch.tensor([[19, 19]])
    expected = bridge(ids)
    with audio.frozen_audio_forward(bridge, ids, _features()):
        actual = bridge(ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(bridge.language_model.embedding._forward_hooks) == 0


@pytest.mark.parametrize("multi_modal_inputs", [{}, _features()])
def test_enabled_audio_requires_adapter(multi_modal_inputs):
    bridge = Bridge()
    setattr(bridge, audio._ENABLED_ATTRIBUTE, True)
    with pytest.raises(ValueError, match="no adapter"):
        with audio.frozen_audio_forward(bridge, torch.tensor([[19, 19]]), multi_modal_inputs):
            pytest.fail("Must reject unconditioned audio before actor execution")


def test_scattered_embedding_boundary_rejected(checkpoint):
    path, hf, _ = checkpoint
    bridge = Bridge()
    bridge.language_model.embedding.scatter_to_sequence_parallel = True
    with pytest.raises(ValueError, match="before embedding sequence-parallel"):
        audio.install_frozen_audio_adapters([bridge], path, hf)


@pytest.mark.parametrize(
    "failure", ["model_error", "not_consumed", "different_ids", "double_embedding", "scattered_output"]
)
def test_hook_failures_are_detected_and_always_cleaned(checkpoint, failure):
    bridge, _ = _installed(checkpoint)
    ids = torch.tensor([[1, 19, 19, 2]])
    with pytest.raises((ValueError, RuntimeError)):
        with audio.frozen_audio_forward(bridge, ids, _features()):
            if failure == "model_error":
                raise RuntimeError("interrupted model")
            if failure == "different_ids":
                bridge(ids.flip(-1))
            if failure == "double_embedding":
                bridge(ids)
                bridge(ids)
            if failure == "scattered_output":
                bridge.language_model.embedding.forward = lambda **kwargs: torch.ones(2, 1, 4)
                bridge(ids)
    assert len(bridge.language_model.embedding._forward_hooks) == 0


@pytest.mark.parametrize(
    "bad_mask", [torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0]]), torch.zeros(1, 9), torch.full((1, 9), 2)]
)
def test_invalid_feature_masks_fail_closed(checkpoint, bad_mask):
    bridge, _ = _installed(checkpoint)
    mm = {**_features(), "feature_attention_mask": bad_mask}
    with pytest.raises(ValueError, match="mask"):
        with audio.frozen_audio_forward(bridge, torch.tensor([[19, 19]]), mm):
            bridge(torch.tensor([[19, 19]]))


def test_missing_features_and_placeholder_mismatch_fail_closed(checkpoint):
    bridge, _ = _installed(checkpoint)
    with pytest.raises(ValueError, match="both input_features"):
        with audio.frozen_audio_forward(bridge, torch.tensor([[19]]), {"input_features": torch.ones(1, 8, 9)}):
            pass
    with pytest.raises(ValueError, match="placeholder count"):
        with audio.frozen_audio_forward(bridge, torch.tensor([[19]]), _features()):
            pass


def test_later_pipeline_stage_does_not_run_encoder():
    stage = SimpleNamespace(pre_process=False)
    setattr(stage, audio._ENABLED_ATTRIBUTE, True)
    with audio.frozen_audio_forward(stage, torch.tensor([[19, 19]]), _features()):
        pass


def test_missing_shards_and_wrong_hidden_dimension_rejected(checkpoint):
    path, hf, _ = checkpoint
    with pytest.raises(ValueError, match="output dimension"):
        audio.FrozenAudioAdapter(path, hf.audio_config, 19, 99)
    (path / "audio.safetensors").unlink()
    with pytest.raises(ValueError, match="missing audio tensors or shards"):
        audio.FrozenAudioAdapter(path, hf.audio_config, 19, 4)


@pytest.mark.parametrize("cp_size,tp_size", [(1, 1), (1, 2), (2, 2), (4, 2)])
@pytest.mark.parametrize("bridge_kind", ["native", "legacy"])
def test_insertion_precedes_installed_bridge_cp_packing_and_tp_scatter(checkpoint, cp_size, tp_size, bridge_kind):
    # Execute the installed bridge's unmodified packing function without its
    # package __init__, which imports unrelated GPU quantization dependencies.
    # Distribution metadata avoids a tests/utils/megatron namespace fixture
    # shadowing the installed package when this runs with other CPU suites.
    source = (
        distribution("megatron-bridge").locate_file("megatron/bridge/models/qwen_vl/modelling_qwen3_vl/utils.py")
        if bridge_kind == "native"
        else distribution("mbridge").locate_file("mbridge/core/util.py")
    )
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "preprocess_packed_seqs"
    )
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "ProcessGroupCollection": SimpleNamespace,
        "PackedSeqParams": SimpleNamespace,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    preprocess_packed_seqs = namespace["preprocess_packed_seqs"]

    bridge, adapter = _installed(checkpoint)
    ids = torch.tensor([[1, 19, 19, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0]])
    mask = ids != 0
    original = bridge.language_model.embedding(input_ids=ids).transpose(0, 1).detach()
    mm = _features()
    expected = original.clone()
    expected[ids == 19] = adapter.encode(mm["input_features"], mm["feature_attention_mask"], ids.device)
    for rank in range(cp_size):
        groups = SimpleNamespace(
            tp=SimpleNamespace(size=lambda: tp_size),
            cp=SimpleNamespace(size=lambda: cp_size, rank=lambda rank=rank: rank),
        )
        with audio.frozen_audio_forward(bridge, ids, mm):
            inserted = bridge.language_model.embedding(input_ids=ids).transpose(0, 1)
        if bridge_kind == "native":
            kwargs = {"pg_collection": groups}
        else:
            namespace["mpu"] = SimpleNamespace(
                get_tensor_model_parallel_world_size=groups.tp.size,
                get_context_parallel_world_size=groups.cp.size,
                get_context_parallel_rank=groups.cp.rank,
            )
            kwargs = {}
        packed, _ = preprocess_packed_seqs(inserted, mask, **kwargs)
        reference, _ = preprocess_packed_seqs(expected, mask, **kwargs)
        torch.testing.assert_close(packed, reference, atol=0, rtol=0)
        # Both bridges scatter the already CP-local SBH sequence across TP.
        for actual, wanted in zip(packed.chunk(tp_size, dim=1), reference.chunk(tp_size, dim=1), strict=True):
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)


def _installed_projector(checkpoint):
    path, hf, _ = checkpoint
    bridge = Bridge()
    adapters = audio.install_trainable_audio_projector_adapters([bridge], path, hf)
    return bridge, adapters[0]


def test_trainable_projector_preserves_original_hf_numerics_and_registers_only_projection(checkpoint):
    path, hf, reference = checkpoint
    bridge, adapter = _installed_projector(checkpoint)
    frozen = audio.FrozenAudioAdapter(path, hf.audio_config, 19, 4)
    mm = _features((30, 117))
    actual = adapter.encode(mm["input_features"], mm["feature_attention_mask"], torch.device("cpu"))
    expected = frozen.encode(mm["input_features"], mm["feature_attention_mask"], torch.device("cpu"))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.requires_grad
    projector = getattr(bridge, audio.AUDIO_PROJECTOR_ATTRIBUTE)
    assert projector is adapter.projector
    assert set(projector.state_dict()) == {"proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias"}
    assert all(parameter.requires_grad for parameter in projector.parameters())
    assert all(parameter.average_gradients_across_tp_domain for parameter in projector.parameters())
    assert all(not parameter.requires_grad for parameter in adapter.tower.parameters())
    assert not any(parameter.is_meta for parameter in projector.parameters())
    assert isinstance(adapter.tower.proj1, torch.nn.Identity)
    assert isinstance(adapter.tower.proj2, torch.nn.Identity)
    for name, value in projector.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name])
    assert not any("ln_post" in name or "conv" in name for name in bridge.state_dict())


def test_trainable_projector_and_language_receive_gradients_encoder_is_immutable(checkpoint):
    bridge, adapter = _installed_projector(checkpoint)
    ids = torch.tensor([[1, 19, 19, 2]])
    projector_before = {name: tensor.clone() for name, tensor in adapter.projector.state_dict().items()}
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-2, weight_decay=0.1)
    with audio.frozen_audio_forward(bridge, ids, _features()):
        loss = bridge(ids)
    encoder_before = {name: tensor.clone() for name, tensor in adapter.tower.state_dict().items()}
    loss.backward()
    assert bridge.head.weight.grad.abs().sum() > 0
    assert bridge.language_model.embedding.word_embeddings.weight.grad[19].abs().sum() == 0
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in adapter.projector.parameters()
    )
    assert all(parameter.grad is None for parameter in adapter.tower.parameters())
    optimizer.step()
    for name, tensor in adapter.projector.state_dict().items():
        assert not torch.equal(tensor, projector_before[name])
    for name, tensor in adapter.tower.state_dict().items():
        torch.testing.assert_close(tensor, encoder_before[name], atol=0, rtol=0)
    assert len(bridge.language_model.embedding._forward_hooks) == 0


def test_trainable_projector_pure_text_has_no_gradient_decay_or_encoder_load(checkpoint):
    bridge, adapter = _installed_projector(checkpoint)
    ids = torch.tensor([[1, 2, 3]])
    expected = bridge(ids).detach()
    before = {name: tensor.clone() for name, tensor in adapter.projector.state_dict().items()}
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-2, weight_decay=0.1)
    with audio.frozen_audio_forward(bridge, ids, {}):
        actual = bridge(ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    assert all(parameter.grad is None for parameter in adapter.projector.parameters())
    optimizer.step()
    for name, tensor in adapter.projector.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
    assert adapter.tower is None


def test_trained_projector_state_round_trip_and_frozen_encoder_offload(checkpoint):
    bridge, adapter = _installed_projector(checkpoint)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-2)
    ids = torch.tensor([[1, 19, 19, 2]])
    with audio.frozen_audio_forward(bridge, ids, _features()):
        bridge(ids).backward()
    optimizer.step()
    restored, restored_adapter = _installed_projector(checkpoint)
    restored.load_state_dict(bridge.state_dict(), strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-2)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    for model in (bridge, restored):
        with audio.frozen_audio_forward(model, ids, _features()):
            output = model(ids)
        if model is bridge:
            expected = output
        else:
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
    for parameter, restored_parameter in zip(bridge.parameters(), restored.parameters(), strict=True):
        assert optimizer.state[parameter].keys() == restored_optimizer.state[restored_parameter].keys()
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(value, restored_optimizer.state[restored_parameter][key])
    adapter.offload()
    assert getattr(bridge, audio.AUDIO_PROJECTOR_ATTRIBUTE) is adapter.projector
    assert next(adapter.tower.parameters()).device.type == "cpu"
    assert all(not parameter.requires_grad for parameter in restored_adapter.tower.parameters())


def test_trainable_projector_installer_rejects_wrapping_scattering_and_duplicate_registration(checkpoint):
    path, hf, _ = checkpoint
    wrapper = torch.nn.Module()
    wrapper.module = Bridge()
    with pytest.raises(ValueError, match="before model wrapping"):
        audio.install_trainable_audio_projector_adapters([wrapper], path, hf)
    bridge = Bridge()
    bridge.language_model.embedding.reduce_scatter_embeddings = True
    with pytest.raises(ValueError, match="before embedding sequence-parallel"):
        audio.install_trainable_audio_projector_adapters([bridge], path, hf)
    installed, _ = _installed_projector(checkpoint)
    with pytest.raises(ValueError, match="already installed"):
        audio.install_trainable_audio_projector_adapters([installed], path, hf)
    later_stage = torch.nn.Module()
    later_stage.pre_process = False
    assert audio.install_trainable_audio_projector_adapters([later_stage], path, hf) == []
    assert not tuple(later_stage.parameters())


def test_trainable_projector_diagnostics_do_not_report_a_frozen_projector(checkpoint, caplog):
    with caplog.at_level(logging.INFO):
        bridge, _ = _installed_projector(checkpoint)
        ids = torch.tensor([[1, 19, 19, 2]])
        with audio.frozen_audio_forward(bridge, ids, _features()):
            bridge(ids)
    messages = [record.message for record in caplog.records if record.name == audio.logger.name]
    assert len(messages) == 3
    assert all("encoder_grad=False projector_grad=True" in message for message in messages)
