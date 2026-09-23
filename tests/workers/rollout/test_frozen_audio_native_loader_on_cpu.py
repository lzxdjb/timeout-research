"""Exercise vLLM's actual audio loader with both TP=2 partitions on CPU."""

import importlib.util
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("rank", [0, 1])
def test_native_audio_loader_restores_each_tp_partition_after_weight_loss(tmp_path, rank):
    # Import inside the test so lightweight suites do not require vLLM.
    omni = pytest.importorskip("vllm.model_executor.models.qwen3_omni_moe_thinker")
    linear = pytest.importorskip("vllm.model_executor.layers.linear")
    parameter = pytest.importorskip("vllm.model_executor.parameter")
    hf = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

    module_path = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/frozen_audio.py"
    spec = importlib.util.spec_from_file_location("native_frozen_audio_test", module_path)
    restoration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restoration)
    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=16,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        encoder_layers=1,
        num_mel_bins=16,
        downsample_hidden_size=4,
        output_dim=8,
        max_source_positions=16,
    )

    # Derive checkpoint tensors independently from the HF reference encoder.
    with torch.device("meta"):
        reference = hf.Qwen3OmniMoeAudioEncoder(config)
    tensors = {
        name: (torch.arange(p.numel()).reshape(p.shape).float() / p.numel() + index).to(torch.bfloat16)
        for index, (name, p) in enumerate(reference.named_parameters())
    }
    save_file({"model.audio_tower." + k: v for k, v in tensors.items()}, str(tmp_path / "model.safetensors"))

    with ExitStack() as stack:
        # Only process-group metadata and unused forward kernels are replaced.
        # Native tensor classes, QKV packing and TP weight loaders remain real.
        for module in [omni, linear, parameter]:
            for name, value in [("get_tensor_model_parallel_world_size", 2), ("get_tensor_model_parallel_rank", rank)]:
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, return_value=value))
        stack.enter_context(patch.object(omni, "MMEncoderAttention", side_effect=lambda **kwargs: torch.nn.Identity()))
        stack.enter_context(patch.object(omni, "_ACTIVATION_REGISTRY", {"gelu": torch.nn.GELU()}))
        model = torch.nn.Module()
        with torch.device("cpu"):
            model.audio_tower = omni.Qwen3OmniMoeAudioEncoder(config)
        model_config = SimpleNamespace(
            model=str(tmp_path),
            hf_config=SimpleNamespace(architectures=["GageQwen3_5OmniMoeForConditionalGeneration"]),
        )
        for step in [0, 2]:
            with torch.no_grad():
                for p in model.audio_tower.parameters():
                    p.fill_(float("nan"))
            restoration.restore_frozen_audio_tower(model, model_config, [], rank=rank, global_steps=step)
            for name, p in model.audio_tower.named_parameters():
                if ".self_attn.qkv." in name:
                    expected = torch.cat(
                        [
                            tensors[name.replace(".self_attn.qkv.", f".self_attn.{projection}_proj.")].chunk(2, dim=0)[
                                rank
                            ]
                            for projection in "qkv"
                        ],
                        dim=0,
                    )
                elif ".fc1." in name:
                    expected = tensors[name].chunk(2, dim=0)[rank]
                elif name.endswith((".self_attn.out_proj.weight", ".fc2.weight")):
                    expected = tensors[name].chunk(2, dim=1)[rank]
                else:
                    expected = tensors[name]
                torch.testing.assert_close(p, expected.to(p.dtype), rtol=0, atol=0, msg=name)
