"""Regression coverage for frozen audio omitted from the actor weight stream."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "frozen_audio", _ROOT / "verl/workers/rollout/vllm_rollout/frozen_audio.py"
)
_AUDIO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIO)


class _Tower(torch.nn.Module):
    """Small TP=2-shaped tower with fused QKV and replicated output weights."""

    def __init__(self, rank=0):
        super().__init__()
        self.rank = rank
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].self_attn = torch.nn.Module()
        qkv = torch.nn.Module()
        qkv.input_size = 4
        qkv.output_sizes = [4, 4, 4]
        qkv.weight = torch.nn.Parameter(torch.zeros(6, 4))
        qkv.bias = torch.nn.Parameter(torch.zeros(6))
        self.layers[0].self_attn.qkv = qkv
        self.proj = torch.nn.Linear(4, 2)
        self.register_buffer("positions", torch.arange(4), persistent=False)

    @torch.no_grad()
    def load_weights(self, weights):
        loaded = set()
        params = dict(self.named_parameters())
        for name, tensor in weights:
            if "_proj." in name:
                projection = next(p for p in "qkv" if f".{p}_proj." in name)
                target = name.replace(f".{projection}_proj.", ".qkv.")
                offset = "qkv".index(projection) * 2
                params[target][offset : offset + 2].copy_(tensor[self.rank * 2 : (self.rank + 1) * 2])
            else:
                target = name
                params[target].copy_(tensor)
            loaded.add(target)
        return loaded


def _model(rank=0):
    model = torch.nn.Module()
    model.audio_tower = _Tower(rank)
    model.language_model = torch.nn.Linear(4, 4)
    return model


def _config(source):
    return SimpleNamespace(
        model=str(source),
        hf_config=SimpleNamespace(architectures=["GageQwen3_5OmniMoeForConditionalGeneration"]),
    )


def _weights():
    result = {}
    for i, projection in enumerate("qkv"):
        prefix = f"model.audio_tower.layers.0.self_attn.{projection}_proj"
        result[prefix + ".weight"] = torch.arange(16).reshape(4, 4).float() + 100 * i
        result[prefix + ".bias"] = torch.arange(4).float() + 10 * i
    result["model.audio_tower.proj.weight"] = torch.full((2, 4), 7.0)
    result["model.audio_tower.proj.bias"] = torch.full((2,), 8.0)
    # Loading this tensor would regress the trained language model.
    result["model.language_model.weight"] = torch.full((4, 4), -999.0)
    return result


def _write_checkpoint(path, weights, indexed):
    path.mkdir(exist_ok=True)
    if indexed:
        first = {k: v for k, v in weights.items() if "q_proj" in k or "proj.bias" in k}
        second = {k: v for k, v in weights.items() if k not in first}
        save_file(first, str(path / "first.safetensors"))
        save_file(second, str(path / "second.safetensors"))
        (path / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {k: "first.safetensors" if k in first else "second.safetensors" for k in weights}}
            )
        )
    else:
        save_file(weights, str(path / "model.safetensors"))


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("indexed", [False, True])
def test_restore_after_initial_and_repeated_weight_loss_preserves_language(tmp_path, rank, indexed):
    weights = _weights()
    _write_checkpoint(tmp_path, weights, indexed)
    model = _model(rank)
    for step in [0, 1, 2]:
        with torch.no_grad():
            for param in model.audio_tower.parameters():
                param.fill_(float("nan"))  # Simulate discarded level-2 parameters.
            model.language_model.weight.fill_(step + 50)
        result = _AUDIO.restore_frozen_audio_tower(model, _config(tmp_path), [], rank=rank, global_steps=step)
        assert result["tensor_count"] == 8
        assert result["parameter_count"] == 4
        expected = torch.cat(
            [weights[f"model.audio_tower.layers.0.self_attn.{p}_proj.weight"][rank * 2 : (rank + 1) * 2] for p in "qkv"]
        )
        torch.testing.assert_close(model.audio_tower.layers[0].self_attn.qkv.weight, expected)
        torch.testing.assert_close(model.audio_tower.proj.bias, torch.full((2,), 8.0))
        torch.testing.assert_close(model.language_model.weight, torch.full((4, 4), float(step + 50)))
        assert all(torch.isfinite(p).all() for p in model.audio_tower.parameters())
        torch.testing.assert_close(model.audio_tower.positions, torch.arange(4))


@pytest.mark.parametrize("indexed", [False, True])
def test_missing_qkv_constituent_fails_before_any_copy(tmp_path, indexed):
    weights = _weights()
    del weights["model.audio_tower.layers.0.self_attn.k_proj.weight"]
    _write_checkpoint(tmp_path, weights, indexed)
    model = _model()
    before = {k: p.clone() for k, p in model.audio_tower.named_parameters()}
    with pytest.raises(RuntimeError, match=r"coverage mismatch.*k_proj.weight"):
        _AUDIO.restore_frozen_audio_tower(model, _config(tmp_path), [])
    for k, p in model.audio_tower.named_parameters():
        torch.testing.assert_close(p, before[k])


def test_bad_global_tp_shape_fails_before_any_copy(tmp_path):
    weights = _weights()
    weights["model.audio_tower.layers.0.self_attn.k_proj.weight"] = torch.zeros(6, 4)
    _write_checkpoint(tmp_path, weights, True)
    model = _model()
    with pytest.raises(RuntimeError, match="Audio tensor shape mismatch"):
        _AUDIO.restore_frozen_audio_tower(model, _config(tmp_path), [])
    assert torch.count_nonzero(model.audio_tower.layers[0].self_attn.qkv.weight) == 0


def test_missing_indexed_tensor_fails(tmp_path):
    weights = _weights()
    _write_checkpoint(tmp_path, weights, True)
    save_file({"unrelated": torch.ones(1)}, str(tmp_path / "first.safetensors"))
    with pytest.raises(RuntimeError, match="is absent from"):
        _AUDIO.restore_frozen_audio_tower(_model(), _config(tmp_path), [])


def test_missing_source_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="local safetensors checkpoint"):
        _AUDIO.restore_frozen_audio_tower(_model(), _config(tmp_path), [])


def test_missing_shard_fails(tmp_path):
    _write_checkpoint(tmp_path, _weights(), True)
    (tmp_path / "first.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        _AUDIO.restore_frozen_audio_tower(_model(), _config(tmp_path), [])


def test_unexpected_checkpoint_audio_key_fails(tmp_path):
    weights = _weights()
    weights["model.audio_tower.unknown.weight"] = torch.ones(1)
    _write_checkpoint(tmp_path, weights, False)
    with pytest.raises(RuntimeError, match=r"unexpected=.*unknown.weight"):
        _AUDIO.restore_frozen_audio_tower(_model(), _config(tmp_path), [])


@pytest.mark.parametrize("fused", [False, True])
def test_complete_actor_audio_takes_precedence_without_reading_source(tmp_path, fused):
    model = _model()
    before = {k: p.clone() for k, p in model.audio_tower.named_parameters()}
    names = set(before) if fused else {_AUDIO.audio_weight_name(k) for k in _weights() if "audio_tower" in k}
    names.add("positions")
    result = _AUDIO.restore_frozen_audio_tower(model, _config(tmp_path / "does-not-exist"), names)
    assert result["source"] == "actor"
    for k, p in model.audio_tower.named_parameters():
        torch.testing.assert_close(p, before[k])


def test_partial_actor_audio_is_not_mixed_with_checkpoint(tmp_path):
    _write_checkpoint(tmp_path, _weights(), False)
    names = {_AUDIO.audio_weight_name(k) for k in _weights() if "audio_tower" in k}
    names.remove("layers.0.self_attn.k_proj.weight")
    with pytest.raises(RuntimeError, match=r"Incomplete synchronized audio tower.*k_proj.weight"):
        _AUDIO.restore_frozen_audio_tower(_model(), _config(tmp_path), names)


def _projector_model(rank=0):
    model = _model(rank)
    del model.audio_tower.proj
    model.audio_tower.proj1 = torch.nn.Linear(4, 4)
    model.audio_tower.proj2 = torch.nn.Linear(4, 2)
    return model


def _projector_weights():
    weights = {name: value for name, value in _weights().items() if not name.startswith("model.audio_tower.proj.")}
    for projection, shape in [("proj1", (4, 4)), ("proj2", (2, 4))]:
        weights[f"model.audio_tower.{projection}.weight"] = torch.full(shape, -999.0)
        weights[f"model.audio_tower.{projection}.bias"] = torch.full(shape[:1], -999.0)
    return weights


def _projector_config(source):
    config = _config(source)
    config.hf_config.verl_audio_projector_trainable = True
    return config


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("indexed", [False, True])
def test_trainable_projector_preserved_across_repeated_frozen_encoder_restoration(tmp_path, rank, indexed):
    weights = _projector_weights()
    _write_checkpoint(tmp_path, weights, indexed)
    model = _projector_model(rank)
    for step in [0, 1, 2]:
        with torch.no_grad():
            for param in model.audio_tower.parameters():
                param.fill_(float("nan"))
            model.language_model.weight.fill_(step + 50)
        actor_weights = {
            name: torch.full_like(param, step + 100.0 + index)
            for index, (name, param) in enumerate(model.audio_tower.named_parameters())
            if name.startswith(("proj1.", "proj2."))
        }
        model.audio_tower.load_weights(actor_weights.items())
        result = _AUDIO.restore_frozen_audio_tower(
            model, _projector_config(tmp_path), actor_weights, rank=rank, global_steps=step
        )
        assert result["tensor_count"] == 6
        assert result["parameter_count"] == 2
        assert result["actor_projector_tensor_count"] == 4
        for leaf in ["weight", "bias"]:
            expected = torch.cat(
                [
                    weights[f"model.audio_tower.layers.0.self_attn.{projection}_proj.{leaf}"][rank * 2 : (rank + 1) * 2]
                    for projection in "qkv"
                ]
            )
            torch.testing.assert_close(getattr(model.audio_tower.layers[0].self_attn.qkv, leaf), expected)
        for name, tensor in actor_weights.items():
            torch.testing.assert_close(dict(model.audio_tower.named_parameters())[name], tensor)
        torch.testing.assert_close(model.language_model.weight, torch.full((4, 4), float(step + 50)))
        assert all(torch.isfinite(param).all() for param in model.audio_tower.parameters())


@pytest.mark.parametrize("missing", [None, "proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias", "all"])
def test_trainable_projector_requires_every_actor_tensor_before_checkpoint_reads(tmp_path, missing):
    model = _projector_model()
    before = {name: param.clone() for name, param in model.audio_tower.named_parameters()}
    names = {"proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias"}
    if missing == "all":
        names.clear()
    elif missing is None:
        names.add("layers.0.self_attn.q_proj.weight")
    else:
        names.remove(missing)
    with pytest.raises(RuntimeError, match="synchronized trainable audio projector"):
        _AUDIO.restore_frozen_audio_tower(model, _projector_config(tmp_path / "no-checkpoint"), names)
    for name, param in model.audio_tower.named_parameters():
        torch.testing.assert_close(param, before[name])


def test_partial_projector_requires_explicit_opt_in(tmp_path):
    names = {"proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias"}
    with pytest.raises(RuntimeError, match="Incomplete synchronized audio tower"):
        _AUDIO.restore_frozen_audio_tower(_projector_model(), _config(tmp_path), names)


def test_trainable_projector_does_not_read_stale_checkpoint_projector(tmp_path):
    weights = _projector_weights()
    # These stale tensors are intentionally unusable. Only encoder tensors
    # should be read when the four projector tensors came from the actor.
    weights["model.audio_tower.proj1.weight"] = torch.full((1,), float("nan"))
    del weights["model.audio_tower.proj2.bias"]
    _write_checkpoint(tmp_path, weights, True)
    model = _projector_model()
    before = {name: param.clone() for name, param in model.audio_tower.named_parameters() if name.startswith("proj")}
    _AUDIO.restore_frozen_audio_tower(model, _projector_config(tmp_path), before)
    for name, tensor in before.items():
        torch.testing.assert_close(dict(model.audio_tower.named_parameters())[name], tensor)


def test_loader_must_report_all_target_parameters(tmp_path):
    _write_checkpoint(tmp_path, _weights(), False)
    model = _model()
    model.audio_tower.load_weights = lambda weights: set()
    with pytest.raises(RuntimeError, match="Audio loader coverage mismatch"):
        _AUDIO.restore_frozen_audio_tower(model, _config(tmp_path), [])


def test_non_audio_model_is_noop(tmp_path):
    assert _AUDIO.restore_frozen_audio_tower(torch.nn.Linear(1, 1), SimpleNamespace(), []) is None


def test_audio_architecture_requires_a_tower(tmp_path):
    with pytest.raises(RuntimeError, match="missing its audio_tower"):
        _AUDIO.restore_frozen_audio_tower(torch.nn.Linear(1, 1), _config(tmp_path), [])


def test_prefix_matching_is_exact():
    assert _AUDIO.audio_weight_name("model.audio_tower.proj.weight") == "proj.weight"
    assert _AUDIO.audio_weight_name("audio_tower.proj.weight") == "proj.weight"
    assert _AUDIO.audio_weight_name("model.language_model.audio_projection.weight") is None


def test_cuda_memory_snapshot_is_safe_without_cuda(caplog):
    caplog.set_level("INFO", logger="frozen_audio")
    snapshot = _AUDIO.cuda_memory_snapshot()
    assert snapshot.get("cuda_available") is False or "memory_query_error" in snapshot
    _AUDIO.log_cuda_memory_snapshot("cpu_test", rank=0, global_steps=7)
    assert "vllm_cuda_memory phase=cpu_test" in caplog.text
