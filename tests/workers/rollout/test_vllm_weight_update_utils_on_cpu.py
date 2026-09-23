# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_weight_update_utils = _load_weight_update_utils()
apply_buffer_updates = _weight_update_utils.apply_buffer_updates
split_buffer_updates = _weight_update_utils.split_buffer_updates


def _load_vllm_rollout_utils():
    """Load vllm_rollout/utils.py with heavyweight deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do not
    leak into other tests; the loaded module keeps working since it binds the
    names it needs at import time.
    """
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        pass

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack
    fake_vllm_utils.resolve_weight_name = lambda model, name, names: name

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.is_fp8_model = lambda config: False
    fake_vllm_quant.load_quanted_weights = lambda weights, runner, is_drafter=False: weights

    fake_platform = types.ModuleType("verl.plugin.platform")
    fake_platform.get_platform = lambda: None

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_quant_utils": fake_vllm_quant,
        "verl.plugin.platform": fake_platform,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyBlock()])


class _FakeVllmConfig:
    def __init__(self, speculative_config=None):
        self.speculative_config = speculative_config


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None):
        self.model = inner_model
        self.vllm_config = _FakeVllmConfig(speculative_config=speculative_config)


def test_split_buffer_updates_routes_registered_buffers():
    model = _ToyModel()
    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32)),
    ]

    param_updates, buffer_updates, named_buffers = split_buffer_updates(model, weights)

    assert [name for name, _ in param_updates] == ["model.layers.0.linear.weight"]
    assert [name for name, _ in buffer_updates] == ["model.layers.0.e_score_correction_bias"]
    assert "model.layers.0.e_score_correction_bias" in named_buffers


def test_apply_buffer_updates_copies_buffer_values():
    model = _ToyModel()
    updates = [("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 1)]

    loaded = apply_buffer_updates(model, updates)

    assert loaded == 1
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    )


def test_apply_buffer_updates_ignores_non_buffer_weights():
    model = _ToyModel()
    weights = [("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32))]

    loaded = apply_buffer_updates(model, weights)

    assert loaded == 0
    assert torch.count_nonzero(model.model.layers[0].e_score_correction_bias) == 0


def test_vllm_update_weights_loads_params_and_buffers():
    model = _ToyModel()
    loaded_param_names = []
    apply_named_buffers = []

    def _fake_load_weights(weights):
        loaded_param_names.extend(name for name, _ in weights)

    model.load_weights = _fake_load_weights

    original_apply_buffer_updates = _vllm_rollout_utils.apply_buffer_updates

    def _spy_apply_buffer_updates(inner_model, buffer_updates, named_buffers=None):
        apply_named_buffers.append(named_buffers)
        return original_apply_buffer_updates(inner_model, buffer_updates, named_buffers=named_buffers)

    _vllm_rollout_utils.apply_buffer_updates = _spy_apply_buffer_updates

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    try:
        worker._update_weights(weights, peft_config=None, base_sync_done=False)
    finally:
        _vllm_rollout_utils.apply_buffer_updates = original_apply_buffer_updates

    assert loaded_param_names == ["model.layers.0.linear.weight"]
    assert apply_named_buffers and apply_named_buffers[0] is not None
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    )


def test_vllm_update_weights_syncs_buffers_to_mtp_drafter():
    """When an MTP drafter is synced, its registered buffers must be updated too."""
    main_model = _ToyModel()
    drafter_model = _ToyModel()
    main_model.load_weights = lambda weights: None
    drafter_model.load_weights = lambda weights: None

    class _SpecConfig:
        method = "mtp"
        draft_model_config = object()

    class _Drafter:
        def __init__(self, m):
            self.model = m

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.drafter = _Drafter(drafter_model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)


class _AudioTower(torch.nn.Linear):
    @torch.no_grad()
    def load_weights(self, weights):
        names = set()
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].copy_(tensor)
            names.add(name)
        return names


def _audio_update_worker(tmp_path, monkeypatch, events, actor_audio=False):
    # Load the real restoration helper without importing the heavyweight
    # rollout package while its IPC transport is replaced by a CPU fake.
    module_name = "verl.workers.rollout.vllm_rollout.frozen_audio"
    spec = importlib.util.spec_from_file_location(
        module_name, _REPO_ROOT / "verl/workers/rollout/vllm_rollout/frozen_audio.py"
    )
    audio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio_module)
    monkeypatch.setitem(sys.modules, module_name, audio_module)
    model = _ToyModel()
    model.audio_tower = _AudioTower(4, 2)
    save_file(
        {
            "model.audio_tower.weight": torch.full((2, 4), 7.0),
            "model.audio_tower.bias": torch.full((2,), 8.0),
            "model.language_model.weight": torch.full((4, 4), -999.0),
        },
        str(tmp_path / "model.safetensors"),
    )
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)
    worker.model_runner.vllm_config.model_config = types.SimpleNamespace(
        model=str(tmp_path),
        hf_config=types.SimpleNamespace(architectures=["GageQwen3_5OmniMoeForConditionalGeneration"]),
    )
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/fake-audio-update.sock"

    @torch.no_grad()
    def load_weights(weights):
        params = dict(model.named_parameters())
        for name, tensor in weights:
            target = name.removeprefix("model.") if name.startswith("model.audio_tower.") else name
            params[target].copy_(tensor)
        events.append("actor_weights_loaded")

    model.load_weights = load_weights

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            first = [("model.layers.0.linear.weight", torch.full((4, 4), 99.0))]
            second = []
            if actor_audio:
                first.append(("model.audio_tower.weight", torch.full((2, 4), 51.0)))
                second.append(("model.audio_tower.bias", torch.full((2,), 52.0)))
            on_bucket_received(first, False)
            events.append("first_ack")
            on_bucket_received(second, True)
            events.append("final_ack")

    receiver_module = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    receiver_module.BucketedWeightReceiver = Receiver
    monkeypatch.setitem(sys.modules, receiver_module.__name__, receiver_module)
    post_module = types.ModuleType("vllm.model_executor.model_loader.utils")

    def post_process(inner_model, config, device):
        assert events[-1] == "final_ack"
        assert torch.isfinite(inner_model.audio_tower.weight).all()
        expected_weight = 51.0 if actor_audio else 7.0
        torch.testing.assert_close(inner_model.audio_tower.weight, torch.full((2, 4), expected_weight))
        events.append("post_process")

    post_module.process_weights_after_loading = post_process
    monkeypatch.setitem(sys.modules, post_module.__name__, post_module)
    return worker, model


def test_audio_restored_after_every_ipc_round_before_post_processing(tmp_path, monkeypatch, caplog):
    events = []
    worker, model = _audio_update_worker(tmp_path, monkeypatch, events)
    for step in [0, 2]:
        with torch.no_grad():
            model.audio_tower.weight.fill_(float("nan"))
            model.audio_tower.bias.fill_(float("nan"))
        worker.update_weights_from_ipc(global_steps=step)
        assert events[-2:] == ["final_ack", "post_process"]
        torch.testing.assert_close(model.model.layers[0].linear.weight, torch.full((4, 4), 99.0))
        torch.testing.assert_close(model.audio_tower.bias, torch.full((2,), 8.0))
    assert "step=2" in caplog.text
    assert caplog.text.count("Restored frozen Omni audio") == 2


def test_audio_load_failure_propagates_after_all_ipc_buckets_acknowledged(tmp_path, monkeypatch):
    events = []
    worker, model = _audio_update_worker(tmp_path, monkeypatch, events)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="local safetensors checkpoint"):
        worker.update_weights_from_ipc(global_steps=4)
    assert events[-1] == "final_ack"
    assert "post_process" not in events


def test_actor_audio_across_multiple_ipc_buckets_is_preserved(tmp_path, monkeypatch):
    events = []
    worker, model = _audio_update_worker(tmp_path, monkeypatch, events, actor_audio=True)
    # A complete actor stream must not need any fallback checkpoint files.
    (tmp_path / "model.safetensors").unlink()
    worker.update_weights_from_ipc(global_steps=6)
    assert events[-1] == "post_process"
    torch.testing.assert_close(model.audio_tower.bias, torch.full((2,), 52.0))
