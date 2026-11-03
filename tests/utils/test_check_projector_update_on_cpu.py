"""Check saved projector update evidence with tiny CPU-only safetensors fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.audio_data.check_projector_update import AUDIO_PREFIX, PROJECTOR_KEYS, check_projector_update


def _state():
    return {
        **{name: torch.ones(2, 2) for name in PROJECTOR_KEYS},
        f"{AUDIO_PREFIX}conv2d1.weight": torch.ones(2, 2),
        f"{AUDIO_PREFIX}ln_post.weight": torch.ones(2),
        f"{AUDIO_PREFIX}ln_post.bias": torch.zeros(2),
        # These tensors must never be read by the verifier.
        "model.language_model.weight": torch.full((2,), float("nan")),
    }


def _write(path, state, indexed):
    path.mkdir()
    if indexed:
        audio = {name: tensor for name, tensor in state.items() if name.startswith(AUDIO_PREFIX)}
        save_file(audio, path / "audio_tower.safetensors")
        weight_map = {name: "audio_tower.safetensors" for name in audio}
        # No text shard exists: accessing it would fail the test.
        weight_map["model.language_model.weight"] = "unread-text-shard.safetensors"
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(state, path / "model.safetensors")
    return path


@pytest.mark.parametrize("source_indexed,saved_indexed", [(False, False), (True, True), (True, False)])
def test_verifier_streams_audio_only_and_accepts_one_changed_projector_element(
    tmp_path, capsys, source_indexed, saved_indexed
):
    before = _state()
    after = _state()
    after[f"{AUDIO_PREFIX}proj1.weight"][0, 0] += 0.25
    source = _write(tmp_path / "source", before, source_indexed)
    saved = _write(tmp_path / "saved", after, saved_indexed)
    assert check_projector_update(source, saved) == 1
    output = capsys.readouterr().out
    assert "changed=1/4 max_abs_change=0.25" in output
    assert "3 frozen encoder tensors are bitwise identical" in output
    assert "PASS:" in output


@pytest.mark.parametrize(
    "damage,error",
    [
        ("unchanged", "Projector did not change"),
        ("encoder", "Frozen encoder tensor changed"),
        ("ln_post", "Frozen encoder tensor changed"),
        ("signed_zero", "Frozen encoder tensor changed"),
        ("missing_encoder", "Audio key schema mismatch"),
        ("missing_projector", "Incomplete audio tower"),
        ("shape", "Audio tensor schema mismatch"),
        ("dtype", "Audio tensor schema mismatch"),
        ("nonfinite", "Nonfinite projector tensor"),
    ],
)
def test_verifier_rejects_missing_updates_encoder_changes_and_bad_schemas(tmp_path, damage, error):
    before = _state()
    after = _state()
    projector = f"{AUDIO_PREFIX}proj1.weight"
    if damage != "unchanged":
        after[projector][0, 0] += 0.25
    if damage == "encoder":
        after[f"{AUDIO_PREFIX}conv2d1.weight"][0, 0] += 1
    elif damage == "ln_post":
        after[f"{AUDIO_PREFIX}ln_post.weight"][0] += 1
    elif damage == "signed_zero":
        after[f"{AUDIO_PREFIX}ln_post.bias"][0] = -0.0
    elif damage == "missing_encoder":
        del after[f"{AUDIO_PREFIX}ln_post.weight"]
    elif damage == "missing_projector":
        del after[projector]
    elif damage == "shape":
        after[projector] = torch.ones(3, 2)
    elif damage == "dtype":
        after[projector] = after[projector].to(torch.bfloat16)
    elif damage == "nonfinite":
        after[projector][0, 0] = float("nan")
    source = _write(tmp_path / "source", before, True)
    saved = _write(tmp_path / "saved", after, True)
    with pytest.raises(ValueError, match=error):
        check_projector_update(source, saved)


def test_cli_returns_failure_for_unchanged_projector(tmp_path):
    source = _write(tmp_path / "source", _state(), True)
    saved = _write(tmp_path / "saved", _state(), False)
    script = Path(__file__).resolve().parents[2] / "scripts/audio_data/check_projector_update.py"
    result = subprocess.run([sys.executable, str(script), str(source), str(saved)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "FAIL: Projector did not change" in result.stderr
