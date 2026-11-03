"""Exercise package-loaded audio OPD datasets across real filter worker boundaries."""

import copy
from types import SimpleNamespace

import dill
import numpy as np
import pytest
import soundfile as sf
from datasets import Dataset
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import create_rl_dataset
from verl.utils.dataset.audio_opd_dataset import AudioOPDMixedDataset
from verl.utils.dataset.rl_dataset import RLHFDataset


class WordTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


class LocalAudioProcessor:
    """Count text/audio tokens and reject anything except the expected decoded clip."""

    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)

    def apply_chat_template(self, messages, **kwargs):
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.extend("<audio>" if item["type"] == "audio" else item["text"] for item in content)
        return " ".join(parts)

    def __call__(self, *, text, images, videos, audio, sampling_rate, return_tensors):
        assert images is None and videos is None
        assert sampling_rate == 16000 and return_tensors == "pt"
        assert len(audio) == 1
        waveform, rate = audio[0]
        assert rate == sampling_rate
        assert waveform.shape == (1600,) and waveform.dtype == np.float32
        np.testing.assert_allclose(waveform, 0.25)
        # Simulate the extra tokens contributed by the audio feature sequence.
        return {"input_ids": [list(range(len(text[0].split()) + 4))]}


@pytest.mark.parametrize("num_proc", [1, 2])
def test_package_loaded_dataset_filters_mixed_rows_in_real_workers(tmp_path, num_proc):
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.full(1600, 0.25, dtype=np.float32), 16000, subtype="FLOAT")
    long_text = " ".join(["word"] * 12)
    rows = [
        {
            "sample_id": "valid_swe",
            "data_source": "swe_agent_verl",
            "prompt": [{"role": "user", "content": "repair this bug"}],
            "audios": None,
        },
        {
            "sample_id": "valid_audio",
            "data_source": "audio_librispeech",
            "prompt": [{"role": "user", "content": "<audio> transcribe"}],
            "audios": [{"audio": str(audio_path)}],
        },
        {
            "sample_id": "overlong_swe",
            "data_source": "swe_agent_verl",
            "prompt": [{"role": "user", "content": long_text}],
            "audios": None,
        },
        {
            "sample_id": "overlong_audio",
            "data_source": "audio_librispeech",
            "prompt": [{"role": "user", "content": "<audio> " + long_text}],
            "audios": [{"audio": str(audio_path)}],
        },
    ]
    parquet_path = tmp_path / "mixed.parquet"
    Dataset.from_list(rows).to_parquet(parquet_path)
    config = OmegaConf.create(
        {
            "custom_cls": {
                "path": "pkg://verl.utils.dataset.audio_opd_dataset",
                "name": "AudioOPDMixedDataset",
            },
            "cache_dir": str(tmp_path / "cache"),
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": num_proc,
            "max_prompt_length": 8,
            "audio_sampling_rate": 16000,
        }
    )
    processor = LocalAudioProcessor()

    dataset = create_rl_dataset([str(parquet_path)], config, processor.tokenizer, processor)

    assert type(dataset) is AudioOPDMixedDataset
    assert dataset.num_workers == num_proc
    assert list(dataset.dataframe["sample_id"]) == ["valid_swe", "valid_audio"]
    assert list(dataset.dataframe["teacher_route"]) == ["swe", "audio"]
    assert list(dataset.dataframe["opd_source"]) == ["swe", "audio_librispeech"]
    assert dataset.dataframe[1]["prompt"] == rows[1]["prompt"]
    assert dataset[1]["raw_prompt"][0]["content"][0] == {"type": "audio", "audio": str(audio_path)}

    # Check the same serialization boundary used by multiprocess: the package
    # class must resolve by reference, preserving RLHFDataset's inherited methods.
    restored = dill.loads(dill.dumps(dataset))
    assert type(restored) is AudioOPDMixedDataset
    assert isinstance(restored, RLHFDataset)
    messages = restored._build_messages(copy.deepcopy(rows[1]), key="prompt")
    assert restored._extract_audio_info(messages) == [str(audio_path)]
    assert callable(restored.maybe_filter_out_long_prompts)
