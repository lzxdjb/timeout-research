# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from verl.models.mcore import model_forward, model_forward_fused, util
from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron import transformer_impl
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
from verl.workers.engine_workers import TrainingWorker


def _batch(keep):
    return TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.ones(3), torch.ones(7)], layout=torch.jagged),
            "train_sample_mask": torch.tensor(keep),
            "advantages": torch.zeros(2, 7),
        },
        batch_size=[2],
    )


def _engine(monkeypatch, *, enabled=True):
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(moe_loss_respects_train_sample_mask=enabled)
    engine.get_data_parallel_group = lambda: None
    engine.is_mp_src_rank_with_outputs = lambda: True
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    return engine


def test_excluded_minibatch_preserves_weights_and_adam_state(monkeypatch):
    engine = _engine(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.1)
    parameter.square().backward()
    optimizer.step()  # Populate momentum: stepping with zero gradients would still change weights.
    before = parameter.detach().clone()
    state_before = deepcopy(optimizer.state[parameter])
    engine.optimizer_zero_grad = Mock(side_effect=lambda: optimizer.zero_grad(set_to_none=False))
    engine.optimizer_step = Mock(side_effect=optimizer.step)
    engine.forward_backward_batch = Mock(side_effect=AssertionError("Excluded batch reached forward"))

    result = engine.train_batch(_batch([False, False]), lambda: None)

    assert result["optimizer_step_skipped"] is True
    assert result["metrics"]["moe_mask/update_skipped"] == 1
    assert result["model_output"] == {}
    engine.optimizer_zero_grad.assert_called_once()
    engine.optimizer_step.assert_not_called()
    engine.forward_backward_batch.assert_not_called()
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    for key, value in state_before.items():
        torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)


@pytest.mark.parametrize("enabled,remote_active", [(False, False), (True, False), (True, True)])
def test_kept_zero_advantage_or_remote_active_batches_still_train(monkeypatch, enabled, remote_active):
    engine = _engine(monkeypatch, enabled=enabled)
    batch = _batch([False, False] if remote_active or not enabled else [True, False])
    if remote_active:
        # The actual distributed reduction is covered separately with Gloo.
        monkeypatch.setattr(transformer_impl, "moe_mask_batch_counts", lambda *a, **k: torch.tensor([1, 4, 7, 20]))
    elif not enabled:
        monkeypatch.setattr(
            transformer_impl, "moe_mask_batch_counts", Mock(side_effect=AssertionError("Opt-out inspected mask"))
        )
    engine.optimizer_zero_grad = Mock()
    engine.optimizer_step = Mock(return_value=0.2)
    engine.forward_backward_batch = Mock(return_value={"loss": [0.0], "metrics": {}, "model_output": {}})

    result = engine.train_batch(batch, lambda: None)

    engine.forward_backward_batch.assert_called_once()
    engine.optimizer_step.assert_called_once()
    assert result["metrics"]["grad_norm"] == 0.2
    assert not result.get("optimizer_step_skipped", False)
    if enabled:
        assert result["metrics"]["moe_mask/active_sequences"] == 1
    else:
        assert not any(key.startswith("moe_mask/") for key in result["metrics"])


@pytest.mark.parametrize("skipped", [[True, True], [False, True], [True, False], [False, False]])
def test_scheduler_advances_once_if_any_minibatch_updated(skipped):
    results = [
        {"loss": [0.0], "metrics": {}, "model_output": {}, "optimizer_step_skipped": skip} for skip in skipped
    ]
    engine = SimpleNamespace(
        train_mode=lambda **kwargs: nullcontext(),
        train_batch=Mock(side_effect=results),
        lr_scheduler_step=Mock(return_value=1e-6),
        is_mp_src_rank_with_outputs=lambda: True,
    )
    worker = SimpleNamespace(
        loss_fn=lambda: None,
        engine=engine,
        engine_config=SimpleNamespace(
            forward_only=False,
            use_dynamic_bsz=False,
            max_token_len_per_gpu=128,
            micro_batch_size_per_gpu=1,
            use_fused_kernels=False,
            moe_loss_respects_train_sample_mask=True,
        ),
        model_config={},
        _postprocess_output=Mock(
            side_effect=lambda output, **kwargs: tu.get_tensordict({}, {"metrics": output["metrics"]})
        ),
    )
    for index in range(2):
        batch = _batch([False, False])
        tu.assign_non_tensor(batch, global_token_num=[3, 7], update_lr_scheduler=index == 1)
        result = TrainingWorker.train_batch(worker, batch)
        assert "optimizer_step_skipped" not in results[index]
        assert worker._postprocess_output.call_args.kwargs["global_token_num"] == (None if skipped[index] else [3, 7])
        if skipped[index]:
            assert tu.get(result, "metrics")["mfu"] == 0.0
    assert engine.lr_scheduler_step.call_count == int(not all(skipped))
    assert ("lr" in tu.get(result, "metrics")) == (not all(skipped))
    assert worker._moe_mask_pending_scheduler_step is False


@pytest.mark.parametrize("forward_mode", ["unfused", "legacy", "hook"])
@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("keep", [None, [True, True], [True, False]])
@pytest.mark.parametrize("vision", [False, True])
def test_forward_wrappers_pass_packed_router_mask(monkeypatch, forward_mode, layout, keep, vision):
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 4)
    monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda: 1)
    captured = {}

    class Model(torch.nn.Module):
        pre_process = True
        post_process = False
        config = SimpleNamespace(fp8=None)
        _verl_fused_forward_mode = forward_mode

        def forward(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros(1)

    ids = torch.nested.as_nested_tensor(
        [torch.ones(19, dtype=torch.long), torch.ones(35, dtype=torch.long)], layout=torch.jagged
    )
    kwargs = dict(
        model=Model(),
        input_ids=ids,
        multi_modal_inputs={},
        cp_layout=layout,
        pad_token_id=0,
        moe_train_sample_mask=None if keep is None else torch.tensor(keep),
    )
    if forward_mode == "unfused":
        model_forward.gptmodel_forward_model_engine(**kwargs, vision_model=vision)
    else:
        model_forward_fused.fused_forward_model_engine(vision_model=vision)(
            **kwargs, labels=ids, temperature=1.0, calculate_entropy=False,
        )
        assert ("output_processor" in captured) == (forward_mode == "hook")
    if vision:
        assert captured["attention_mask"].sum(1).tolist() == [19, 35]
    if keep is None or all(keep):
        assert "padding_mask" not in captured
    else:
        lengths = (32, 48) if layout == "zigzag" else (24, 40)
        segments = [torch.zeros(lengths[0], dtype=torch.bool), torch.ones(lengths[1], dtype=torch.bool)]
        expected = (
            torch.cat([torch.cat([s.chunk(8)[1], s.chunk(8)[6]]) for s in segments])
            if layout == "zigzag" else torch.cat(segments).chunk(4)[1]
        )
        assert torch.equal(captured["padding_mask"], expected.unsqueeze(0))


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("enabled,training", [(False, True), (True, True), (True, False)])
def test_engine_passes_final_sequence_mask_only_during_enabled_training(monkeypatch, fused, enabled, training):
    engine = _engine(monkeypatch, enabled=enabled)
    engine.engine_config.use_fused_kernels = fused
    engine.engine_config.use_remove_padding = True
    engine.engine_config.dynamic_context_parallel = False
    engine.engine_config.pad_to_length = False
    engine.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        mtp=SimpleNamespace(enable=False),
    )
    engine.tf_config = SimpleNamespace(calculate_per_token_loss=False)
    engine.enable_routing_replay = False
    engine.prepare_model_inputs = lambda batch: {
        "input_ids": batch["input_ids"],
        "attention_mask": None,
        "multi_modal_inputs": {},
        "loss_mask": torch.ones(2, 7),
    }
    for method in ("is_replay_backward_action", "is_replay_forward_action", "is_r2_record_action"):
        monkeypatch.setattr(transformer_impl.RouterReplayHelper, method, lambda *args: False)
    forward = Mock(return_value=torch.zeros(1))
    import verl.models.mcore

    monkeypatch.setattr(verl.models.mcore, "get_mcore_forward_fused_model_engine_fn", lambda config: forward)
    monkeypatch.setattr(verl.models.mcore, "get_mcore_engine_forward_fn", lambda config: forward)
    batch = _batch([True, False])
    batch["temperature"] = torch.ones(2)
    with torch.set_grad_enabled(training):
        engine.forward_step(iter([batch]), SimpleNamespace(config=SimpleNamespace()), None, lambda: None)
    kwargs = forward.call_args.kwargs
    if enabled and training:
        assert kwargs["moe_train_sample_mask"].tolist() == [True, False]
    else:
        assert "moe_train_sample_mask" not in kwargs
