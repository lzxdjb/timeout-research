"""Real media/features/tokenization and checkpoint compatibility without vLLM."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

from verl.utils.tokenizer import build_multimodal_processor_inputs, hf_processor
from verl.utils.tokenizer.audio_processor import load_audio_feature_extractor, load_audio_waveform


@pytest.fixture
def checkpoint(tmp_path):
    tokens = [
        "[UNK]",
        "[PAD]",
        "<|audio_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "hello",
    ]
    backend = Tokenizer(models.WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", additional_special_tokens=tokens[2:7]
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor()
    )
    processor.save_pretrained(tmp_path)
    config = {
        "model_type": "qwen3_5_moe",
        "verl_omni_audio": True,
        "audio_token_id": 2,
        "audio_config": {"num_mel_bins": 128},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    # Match the selected global_step_18 hybrid export: no separate extractor
    # file, and explicit audio fields stored beside the image settings.
    file = tmp_path / "processor_config.json"
    saved = json.loads(file.read_text())
    saved["image_processor"].update(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        chunk_length=30,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=True,
    )
    file.write_text(json.dumps(saved))
    return tmp_path


@pytest.mark.parametrize("extension,rate", [("wav", 8000), ("flac", 16000)])
def test_real_local_media_produces_features_and_matching_placeholders(checkpoint, extension, rate):
    # Include stereo and a partial hop so loading/resampling/padding are exercised.
    samples = np.sin(np.arange(rate + 1) * 0.03).astype("float32")
    path = checkpoint / f"clip.{extension}"
    sf.write(path, np.stack([samples, samples * 0.5], axis=1), rate)
    processor = hf_processor(checkpoint, enable_audio_processor=True, local_files_only=True)
    result = build_multimodal_processor_inputs(processor, text=["hello <|audio_pad|>"], audio=[str(path)])
    direct = processor.extract_audio_features([(load_audio_waveform(path, target_sampling_rate=16000), 16000)])
    torch.testing.assert_close(result["input_features"], direct["input_features"], atol=0, rtol=0)
    torch.testing.assert_close(result["feature_attention_mask"], direct["feature_attention_mask"])
    assert result["input_features"].shape[:2] == (1, 128)
    frames = int(result["feature_attention_mask"].sum())
    remainder = frames % 100
    expected = (((remainder - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1 + frames // 100 * 13
    assert int((result["input_ids"] == processor.audio_token_id).sum()) == expected
    assert hasattr(processor, "get_rope_index")


def test_save_roundtrip_keeps_default_processor_and_opt_in_features(checkpoint, tmp_path):
    plain = hf_processor(checkpoint, local_files_only=True)
    processor = hf_processor(checkpoint, enable_audio_processor=True, local_files_only=True)
    text = {"text": ["hello"], "return_tensors": "pt"}
    torch.testing.assert_close(plain(**text)["input_ids"], processor(**text)["input_ids"])
    destination = tmp_path / "export"
    processor.save_pretrained(destination)
    (destination / "config.json").write_text((checkpoint / "config.json").read_text())
    saved = json.loads((destination / "processor_config.json").read_text())
    assert saved["processor_class"] == "Qwen3VLProcessor"
    assert (destination / "preprocessor_config.json").is_file()
    default_reloaded = hf_processor(destination, local_files_only=True)
    enabled_reloaded = hf_processor(destination, enable_audio_processor=True, local_files_only=True)
    assert type(default_reloaded).__name__ == "Qwen3VLProcessor"
    assert type(enabled_reloaded).__name__ == "AudioQwen3VLProcessor"
    waveform = (np.sin(np.arange(16000) * 0.03).astype("float32"), 16000)
    before = processor.extract_audio_features([waveform])
    after = enabled_reloaded.extract_audio_features([waveform])
    torch.testing.assert_close(before["input_features"], after["input_features"], atol=0, rtol=0)
    torch.testing.assert_close(default_reloaded(**text)["input_ids"], plain(**text)["input_ids"])


def test_real_audio_postprocess_builds_required_rope_types_without_processor_marker(checkpoint):
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    processor = hf_processor(checkpoint, enable_audio_processor=True, local_files_only=True)
    waveform = (np.sin(np.arange(16000) * 0.03).astype("float32"), 16000)
    prompt = build_multimodal_processor_inputs(processor, text=["hello <|audio_pad|>"], audio=[waveform])
    # The rollout returns already-expanded prompt IDs plus a generated answer,
    # and postprocess pads the complete sequence before building RoPE inputs.
    ids = torch.cat(
        (torch.tensor([[processor.tokenizer.pad_token_id]]), prompt["input_ids"], torch.tensor([[7]])), dim=1
    )
    attention_mask = torch.ones_like(ids)
    attention_mask[:, 0] = 0
    worker = SimpleNamespace(processor=processor, tokenizer=processor.tokenizer)
    output = SimpleNamespace(multi_modal_data={"audios": [waveform]}, mm_processor_kwargs={"sampling_rate": 16000})
    mm_inputs = AgentLoopWorker._compute_multi_modal_inputs(worker, output, ids)
    assert set(mm_inputs) == {"input_features", "feature_attention_mask"}
    torch.testing.assert_close(mm_inputs["input_features"], prompt["input_features"], atol=0, rtol=0)
    positions = AgentLoopWorker._compute_position_ids(worker, ids, attention_mask, mm_inputs)
    assert positions.shape == (1, 4, ids.shape[-1])
    expected = torch.arange(int(attention_mask.sum()))
    torch.testing.assert_close(positions[0, :, 1:], expected.expand(4, -1))
    assert "mm_token_type_ids" not in mm_inputs


def test_incomplete_saved_audio_configuration_fails_closed(checkpoint):
    saved_path = checkpoint / "processor_config.json"
    saved = json.loads(saved_path.read_text())
    del saved["image_processor"]["sampling_rate"]
    saved_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="complete saved"):
        load_audio_feature_extractor(checkpoint, local_files_only=True)
    with pytest.raises(RuntimeError, match="explicitly enabled"):
        hf_processor(checkpoint, enable_audio_processor=True, local_files_only=True)


@pytest.mark.parametrize("value", [np.ones(4), (np.ones(4), 0), (np.array([np.nan]), 16000)])
def test_array_requires_finite_waveform_and_explicit_valid_rate(value):
    with pytest.raises(ValueError):
        load_audio_waveform(value, target_sampling_rate=16000)


def test_mismatched_checkpoint_audio_schema_rejected(checkpoint):
    file = checkpoint / "config.json"
    config = json.loads(file.read_text())
    config["audio_token_id"] = 999
    file.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="explicitly enabled"):
        hf_processor(checkpoint, enable_audio_processor=True, local_files_only=True)
