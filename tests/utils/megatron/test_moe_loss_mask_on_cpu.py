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

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe import router as router_module
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, compute_routing_scores_for_aux_loss
from megatron.core.transformer.moe.router import TopKRouter
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from verl.models.mcore import util
from verl.utils.megatron.moe_loss_mask import (
    _apply_masked_aux_loss,
    install_moe_loss_mask_support,
    moe_mask_batch_counts,
    packed_moe_router_mask,
    validate_moe_loss_mask_config,
)


@pytest.fixture(autouse=True)
def isolate_native_auxiliary_state(monkeypatch):
    previous = MoEAuxLossAutoScaler.main_loss_backward_scale
    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0)
    monkeypatch.setattr(router_module, "get_moe_metrics_tracker", lambda: SimpleNamespace(record=Mock()))
    yield
    MoEAuxLossAutoScaler.main_loss_backward_scale = previous


@pytest.mark.parametrize("layout,bucket", [("zigzag", None), ("contiguous", None), ("contiguous", 128)])
def test_mask_tracks_real_packing_cp4_tp2_and_alignment_padding(monkeypatch, layout, bucket):
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 4)
    ids = torch.nested.as_nested_tensor(
        [torch.full((length,), i + 1, dtype=torch.long) for i, length in enumerate([3, 19, 35])],
        layout=torch.jagged,
    )
    keep = torch.tensor([True, False, True])
    for rank in range(4):
        monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        local_ids, packed, _ = util.preprocess_thd_engine(ids, cp_layout=layout, pad_to_length_bucket=bucket)
        mask = packed_moe_router_mask(
            keep, packed.cu_seqlens_q_padded, cp_size=4, cp_rank=rank, cp_layout=layout
        )
        # Independent construction from each global segment's two CP chunks.
        segments = [
            torch.full((n,), not flag)
            for n, flag in zip(packed.cu_seqlens_q_padded.diff().tolist(), keep, strict=True)
        ]
        if layout == "zigzag":
            expected = torch.cat([torch.cat([s.chunk(8)[rank], s.chunk(8)[7 - rank]]) for s in segments])
        else:
            expected = torch.cat(segments).chunk(4)[rank]
        assert mask.shape == local_ids.shape
        assert torch.equal(mask[0], expected)
        assert mask[local_ids == 2].all()
        assert not mask[(local_ids == 1) | (local_ids == 3)].any()
        # TP sequence parallelism scatters the same contiguous token dimension.
        for ids_tp, mask_tp in zip(local_ids.chunk(2, dim=1), mask.chunk(2, dim=1), strict=True):
            assert mask_tp[ids_tp == 2].all()
            assert not mask_tp[(ids_tp == 1) | (ids_tp == 3)].any()


def test_all_kept_and_absent_masks_preserve_existing_padding_policy():
    offsets = torch.tensor([0, 8, 16])
    existing = torch.tensor([[True, False, False, False]])
    for keep in (None, torch.ones(2, dtype=torch.bool)):
        assert packed_moe_router_mask(keep, offsets, cp_size=4, cp_rank=0, existing_mask=existing) is existing
        assert packed_moe_router_mask(keep, offsets, cp_size=4, cp_rank=0) is None
    combined = packed_moe_router_mask(
        torch.tensor([True, False]), offsets, cp_size=4, cp_rank=0, existing_mask=existing
    )
    assert combined.tolist() == [[True, False, True, True]]


def test_mask_matches_qwen35_bridge_repacking_cp4_tp2(monkeypatch):
    from mbridge.core.util import preprocess_packed_seqs

    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 4)
    ids = torch.nested.as_nested_tensor(
        [torch.ones(19, dtype=torch.long), torch.full((35,), 2, dtype=torch.long)], layout=torch.jagged
    )
    padded = ids.to_padded_tensor(0)
    attention = padded != 0
    for rank in range(4):
        monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda rank=rank: rank)
        local_ids, packed, _ = util.preprocess_thd_engine(ids)
        bridge_ids, bridge_packed = preprocess_packed_seqs(padded, attention)
        assert torch.equal(bridge_ids, local_ids)
        assert torch.equal(bridge_packed.cu_seqlens_q_padded, packed.cu_seqlens_q_padded)
        mask = packed_moe_router_mask(
            torch.tensor([True, False]), packed.cu_seqlens_q_padded, cp_size=4, cp_rank=rank
        )
        assert mask[bridge_ids == 2].all()
        assert not mask[bridge_ids == 1].any()


@pytest.mark.parametrize("keep", [torch.tensor([1]), torch.tensor([0.5, 1.0]), torch.tensor([float("nan"), 0])])
def test_invalid_sequence_masks_fail(keep):
    with pytest.raises(ValueError, match="train_sample_mask"):
        packed_moe_router_mask(keep, torch.tensor([0, 8, 16]), cp_size=4, cp_rank=0)


def _z_router():
    return SimpleNamespace(
        config=SimpleNamespace(moe_z_loss_coeff=0.001, num_layers=1, mtp_num_layers=None),
        training=True,
        is_mtp_layer=False,
        calculate_per_token_loss=False,
        tp_cp_group=SimpleNamespace(size=lambda: 1),
        layer_number=1,
    )


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("excluded", [[False] * 6, [False, False, True, True, False, True], [True] * 6])
def test_native_z_loss_gradients_and_activation_recomputation(recompute, excluded):
    torch.manual_seed(21)
    logits = torch.randn(6, 8, requires_grad=True)
    mask = torch.tensor(excluded)
    router = _z_router()
    fn = lambda x, m: TopKRouter.apply_z_loss(router, x, padding_mask=m)
    output = checkpoint(fn, logits, mask, use_reentrant=True) if recompute else fn(logits, mask)
    # No policy gradient: this isolates the actual native auxiliary backward.
    (output * 0).sum().backward()
    reference = logits.detach().clone().requires_grad_()
    expected_loss = (reference.logsumexp(-1).square() * ~mask).sum() / (~mask).sum().clamp_min(1) * 0.001
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[mask]) == 0
    if (~mask).any():
        assert logits.grad[~mask].norm() > 0  # Eligible zero-advantage sequences retain z-loss.


def _balanced_loss(monkeypatch, logits, mask, remote_counts=None):
    group = SimpleNamespace(size=lambda: 1)
    monkeypatch.setattr(
        moe_utils,
        "reduce_from_tensor_model_parallel_region",
        lambda value, group: value if remote_counts is None else value + remote_counts,
    )
    groups = SimpleNamespace(
        loss_reduce_groups=(group,),
        metric_reduce_group=group,
        metric_avg_group=None,
        metric_needs_dp_avg=False,
        metric_pre_reduce_groups=None,
    )
    attached = {}

    def attach(probs, coeff, loss, *args, **kwargs):
        attached.update(loss=loss.detach(), count=kwargs["aux_loss_scale_num_tokens"])
        return MoEAuxLossAutoScaler.apply(probs, loss)

    router = SimpleNamespace(
        topk=2,
        config=SimpleNamespace(num_moe_experts=8),
        get_aux_loss_coeff=lambda name: 0.01,
        _get_aux_loss_groups=lambda packed: groups,
        attach_and_log_load_balancing_loss=attach,
    )
    routes, scores = compute_routing_scores_for_aux_loss(logits, 2, "softmax", padding_mask=mask)
    out = _apply_masked_aux_loss(router, logits.sigmoid(), scores, routes, with_padding_mask=True)
    (out * 0).sum().backward()
    return attached


def test_balancing_loss_is_independent_of_excluded_logits(monkeypatch):
    torch.manual_seed(9)
    a = torch.randn(6, 8, requires_grad=True)
    b = a.detach().clone()
    mask = torch.tensor([False, True, True, False, False, True])
    b[mask] = torch.randn_like(b[mask]) * 100
    b.requires_grad_()
    first = _balanced_loss(monkeypatch, a, mask)
    second = _balanced_loss(monkeypatch, b, mask)
    torch.testing.assert_close(first["loss"], second["loss"])
    torch.testing.assert_close(a.grad[~mask], b.grad[~mask])
    assert torch.count_nonzero(a.grad[mask]) == 0
    assert a.grad[~mask].norm() > 0


@pytest.mark.parametrize("remote", [False, True])
def test_balancing_loss_empty_local_and_empty_global_domains(monkeypatch, remote):
    logits = torch.randn(6, 8, requires_grad=True)
    counts = torch.tensor([2, 2, 2, 2, 0, 0, 0, 0]) if remote else None
    result = _balanced_loss(monkeypatch, logits, torch.ones(6, dtype=torch.bool), remote_counts=counts)
    assert result["count"].item() == (4 if remote else 0)
    assert result["loss"].item() == 0
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


def test_unmasked_balancing_uses_original_implementation():
    original = Mock(return_value="unchanged")
    router = SimpleNamespace(_verl_original_aux_loss=original)
    assert _apply_masked_aux_loss(router, "probs", "scores", "routes") == "unchanged"
    original.assert_called_once()


def _supported_configs():
    engine = SimpleNamespace(
        moe_loss_respects_train_sample_mask=True,
        dynamic_context_parallel=False,
        use_remove_padding=True,
        pipeline_model_parallel_size=1,
    )
    model = SimpleNamespace(model_type="language_model", mtp=SimpleNamespace(enable=False))
    tf = SimpleNamespace(
        calculate_per_token_loss=False,
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_router_load_balancing_type="none",
    )
    return engine, model, tf


@pytest.mark.parametrize(
    "where,key,value",
    [
        (0, "dynamic_context_parallel", True),
        (0, "use_remove_padding", False),
        (0, "pipeline_model_parallel_size", 2),
        (2, "calculate_per_token_loss", True),
        (2, "moe_token_dispatcher_type", "flex"),
        (2, "moe_router_load_balancing_type", "seq_aux_loss"),
        (2, "moe_router_enable_expert_bias", True),
    ],
)
def test_unsupported_configuration_fails_explicitly(where, key, value):
    configs = _supported_configs()
    validate_moe_loss_mask_config(*configs)
    setattr(configs[where], key, value)
    with pytest.raises(ValueError, match="MoE train-sample masking"):
        validate_moe_loss_mask_config(*configs)
    configs[0].moe_loss_respects_train_sample_mask = False
    validate_moe_loss_mask_config(*configs)  # Opt-out is unaffected.


@pytest.mark.parametrize("bucket,fp8", [(True, None), (False, "e4m3")])
def test_vlm_repacking_with_different_padding_is_rejected(bucket, fp8):
    engine, model, tf = _supported_configs()
    model.hf_config = SimpleNamespace(vision_config={})
    engine.pad_to_length = bucket
    tf.fp8 = fp8
    with pytest.raises(ValueError, match="VLM repacking"):
        validate_moe_loss_mask_config(engine, model, tf)


def test_batch_counts_do_not_use_reward_or_advantage():
    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.ones(3), torch.ones(7)], layout=torch.jagged),
            "train_sample_mask": torch.tensor([True, False]),
            "advantages": torch.zeros(2, 7),
        },
        batch_size=[2],
    )
    counts = moe_mask_batch_counts(data, device="cpu", dp_group=None)
    assert counts.tolist() == [1, 2, 3, 10]


@pytest.mark.parametrize("aux_coeff", [0.0, 0.01])
def test_install_is_actor_local_and_idempotent(aux_coeff):
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.get_aux_loss_coeff = lambda name: aux_coeff
    original_class_method = TopKRouter._apply_aux_loss
    install_moe_loss_mask_support([router])
    install_moe_loss_mask_support([router])
    assert TopKRouter._apply_aux_loss is original_class_method
    if aux_coeff:
        assert router._apply_aux_loss.__func__ is _apply_masked_aux_loss
        assert router._verl_original_aux_loss.__func__ is original_class_method
    else:
        assert router._apply_aux_loss.__func__ is original_class_method


def _distributed_counts_worker(rank, init_method):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        data = TensorDict(
            {
                "input_ids": torch.nested.as_nested_tensor([torch.ones(3), torch.ones(7)], layout=torch.jagged),
                "train_sample_mask": torch.tensor([rank == 1, False]),
            },
            batch_size=[2],
        )
        counts = moe_mask_batch_counts(data, device="cpu", dp_group=torch.distributed.group.WORLD)
        assert counts.tolist() == [1, 4, 3, 20]  # Neither rank may skip, including the locally empty rank.
        data["train_sample_mask"].zero_()
        counts = moe_mask_batch_counts(data, device="cpu", dp_group=torch.distributed.group.WORLD)
        assert counts.tolist() == [0, 4, 0, 20]  # Both ranks must skip.
        del data["train_sample_mask"]
        counts = moe_mask_batch_counts(data, device="cpu", dp_group=torch.distributed.group.WORLD)
        assert counts.tolist() == [4, 4, 20, 20]
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_counts_agree_for_local_and_global_exclusions(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_counts_worker, args=(f"file://{tmp_path / 'gloo_init'}",), nprocs=2, join=True
    )
