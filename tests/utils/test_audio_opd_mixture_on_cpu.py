"""Opt-in routing and real StatefulDataLoader resume for source-quota OPD."""

import asyncio
import copy
from collections import Counter
from types import SimpleNamespace

import pytest
from datasets import Dataset
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from scripts.audio_data.multi_teacher_config import hydra_literal, teacher_config
from verl.utils.dataset.audio_opd_dataset import AudioOPDMixedDataset, route_audio_opd_row
from verl.utils.dataset.audio_opd_sampler import AudioOPDQuotaSampler
from verl.utils.dataset.rl_dataset import collate_fn


class Pool:
    def __init__(self):
        self.dataframe = Dataset.from_dict({"opd_source": [s for s in ("swe", "en", "zh", "qa") for _ in range(100)]})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return {"index": index, "source": self.dataframe[index]["opd_source"]}


SCHEDULE = [
    {"steps": 20, "quotas": {"swe": 16, "en": 18, "zh": 18, "qa": 12}},
    {"steps": 60, "quotas": {"swe": 32, "en": 12, "zh": 12, "qa": 8}},
    {"steps": 20, "quotas": {"swe": 48, "en": 6, "zh": 6, "qa": 4}},
]


@pytest.mark.parametrize("step,phase", [(1, 0), (20, 0), (21, 1), (80, 1), (81, 2), (100, 2)])
def test_exact_phase_source_quotas(step, phase):
    pool = Pool()
    sampler = AudioOPDQuotaSampler(pool, OmegaConf.create(SCHEDULE), 64, 17)
    sampler.set_step(step)
    stream = iter(sampler)
    draws = [next(stream) for _ in range(64)]
    assert Counter(pool[index]["source"] for index in draws) == SCHEDULE[phase]["quotas"]
    assert len(set(draws)) == 64
    assert dict(sampler.draw_counts) == SCHEDULE[phase]["quotas"]


def test_loader_resume_mid_quota_matches_uninterrupted_across_phase_change():
    pool = Pool()
    sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 64, 42)
    loader = StatefulDataLoader(pool, batch_size=1, sampler=sampler, num_workers=0)
    stream = iter(loader)
    for _ in range(13):
        next(stream)
    saved = copy.deepcopy(loader.state_dict())
    expected = [next(stream)["index"].item() for _ in range(70)]
    resumed_sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 64, 42)
    resumed_loader = StatefulDataLoader(pool, batch_size=1, sampler=resumed_sampler, num_workers=0)
    resumed_loader.load_state_dict(saved)
    resumed = iter(resumed_loader)
    actual = [next(resumed)["index"].item() for _ in range(70)]
    assert actual == expected
    sampler.set_step(21)
    resumed_sampler.set_step(21)
    assert [next(stream)["index"].item() for _ in range(64)] == [next(resumed)["index"].item() for _ in range(64)]


@pytest.mark.parametrize("fetch_size", [1, 64])
def test_real_trainer_fetch_resumes_current_phase_after_lazy_sampler_restore(fetch_size):
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    class PromptPool(Pool):
        def __getitem__(self, index):
            return {**super().__getitem__(index), "raw_prompt": [{"role": "user", "content": "prompt"}]}

    pool = PromptPool()
    sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 64, 42)
    loader = StatefulDataLoader(pool, batch_size=fetch_size, sampler=sampler, num_workers=0, collate_fn=collate_fn)
    sampler.set_step(20)
    stream = iter(loader)
    for _ in range(64 // fetch_size):
        next(stream)
    saved = copy.deepcopy(loader.state_dict())
    sampler.set_step(21)
    expected = [index for _ in range(64 // fetch_size) for index in next(stream)["index"]]

    resumed_sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 64, 42)
    resumed_loader = StatefulDataLoader(
        pool, batch_size=fetch_size, sampler=resumed_sampler, num_workers=0, collate_fn=collate_fn
    )
    resumed_loader.load_state_dict(saved)
    trainer = SimpleNamespace(
        config=OmegaConf.create({"data": {"opd_source_mix": SCHEDULE}}),
        global_steps=21,
        train_dataloader=resumed_loader,
        train_dataloader_it=None,
    )
    actual = [index for _ in range(64 // fetch_size) for index in PPOTrainer._fetch_one_gen_batch(trainer)["index"]]
    assert actual == expected
    assert Counter(pool[index]["source"] for index in actual) == SCHEDULE[1]["quotas"]
    assert resumed_sampler.step == 21


def test_each_source_cycles_only_after_consuming_its_pool():
    sampler = AudioOPDQuotaSampler(Pool(), SCHEDULE, 64)
    assert len({sampler._draw("swe") for _ in range(100)}) == 100
    assert sampler._draw("swe") in sampler.groups["swe"]


def test_partial_quota_phase_change_never_consumes_unrequested_source_rows():
    pool = Pool()
    sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 64, 42)
    initial_orders = copy.deepcopy(sampler.orders)
    stream = iter(sampler)
    first = [next(stream) for _ in range(13)]
    counts = Counter(pool[index]["source"] for index in first)
    assert sum(sampler.cursors.values()) == 13
    assert sampler.state_dict()["version"] == 2
    sampler.set_step(21)
    following = [next(stream) for _ in range(64)]
    assert Counter(pool[index]["source"] for index in following) == SCHEDULE[1]["quotas"]
    for source in sampler.groups:
        emitted = [index for index in first + following if pool[index]["source"] == source]
        assert emitted == initial_orders[source][: len(emitted)]
        assert sampler.cursors[source] == counts[source] + SCHEDULE[1]["quotas"][source]


def test_changed_curriculum_cannot_resume():
    pool = Pool()
    saved = AudioOPDQuotaSampler(pool, SCHEDULE, 64).state_dict()
    other = copy.deepcopy(SCHEDULE)
    other[0]["steps"] = 19
    with pytest.raises(ValueError, match="different data"):
        AudioOPDQuotaSampler(pool, other, 64).load_state_dict(saved)


@pytest.mark.parametrize(
    "schedule",
    [
        [],
        [{"steps": 1, "quotas": {"swe": 63}}],
        [{"steps": 1, "quotas": {"missing": 64}}],
        [{"steps": 0, "quotas": {"swe": 64}}],
    ],
)
def test_bad_curricula_fail_before_training(schedule):
    with pytest.raises(ValueError):
        AudioOPDQuotaSampler(Pool(), schedule, 64)


def test_audio_metadata_never_changes_prompt_reference_or_original_row():
    row = {
        "data_source": "audio_aishell1",
        "prompt": [{"role": "user", "content": "<audio>"}],
        "audios": [{"audio": "/clip.wav"}],
        "reward_model": {"ground_truth": "unchanged"},
    }
    before = copy.deepcopy(row)
    fields = route_audio_opd_row(row)
    assert row == before
    assert fields["teacher_route"] == "audio"
    assert fields["agent_name"] == "single_turn_agent"
    assert fields["opd_source"] == "audio_aishell1"
    assert not {"prompt", "reward_model", "extra_info"} & fields.keys()


def test_swe_task_identity_and_source_remain_unchanged():
    row = {"data_source": "swe_agent_verl", "extra_info": {"index": 57}, "agent_name": "swe_tool_agent"}
    fields = route_audio_opd_row(row)
    assert fields["teacher_route"] == "swe"
    assert fields["opd_source"] == "swe"
    assert "data_source" not in fields and "extra_info" not in fields


def test_mixed_dataset_loads_original_local_audio_as_mono_waveform_for_student_and_rollout(tmp_path):
    import numpy as np
    import soundfile as sf

    path = tmp_path / "stereo8k.wav"
    sf.write(path, np.stack([np.full(800, 0.25), np.full(800, 0.5)], axis=-1), 8000, subtype="FLOAT")
    messages = [{"role": "user", "content": [{"type": "audio", "audio": str(path)}]}]
    original = copy.deepcopy(messages)
    images, videos, audio = asyncio.run(
        AudioOPDMixedDataset.process_multi_modal_info(messages, 14, OmegaConf.create({"audio_sampling_rate": 16000}))
    )
    assert images is None and videos is None
    assert messages == original
    waveform, sampling_rate = audio[0]
    assert sampling_rate == 16000
    assert waveform.shape == (1600,) and waveform.dtype == np.float32
    assert np.isfinite(waveform).all()
    np.testing.assert_allclose(waveform[200:-200], 0.375, atol=0.001)


@pytest.mark.parametrize(
    "row",
    [
        {"data_source": "audio_x"},
        {"data_source": "swe_agent_verl", "teacher_route": "audio"},
        {"data_source": "swe_agent_verl", "agent_name": "other"},
    ],
)
def test_invalid_routing_fails_closed(row):
    with pytest.raises(ValueError):
        route_audio_opd_row(row)


def test_launcher_values_parse_as_hydra_objects_instead_of_strings():
    parser = OverridesParser.create()
    literal = hydra_literal(SCHEDULE)
    assert parser.parse_override("+data.opd_source_mix=" + literal).value() == SCHEDULE
    env = {
        "TEACHER_BASE_URL": "http://coding:18090",
        "TEACHER_MODEL_NAME": "served",
        "TEACHER_MAX_MODEL_LEN": "262144",
        "TEACHER_MAX_CONCURRENT_REQUESTS": "4",
        "MAX_PROMPT_LENGTH": "8192",
        "MAX_RESPONSE_LENGTH": "253951",
        "AUDIO_TEACHER_BASE_URL": "http://audio:18090",
        "AUDIO_TEACHER_MODEL_NAME": "served",
        "AUDIO_TEACHER_MAX_MODEL_LEN": "65536",
        "AUDIO_TEACHER_MAX_CONCURRENT_REQUESTS": "2",
        "AUDIO_TEACHER_TOKENIZER_PATH": "/audio/tokenizer",
    }
    for route in ("swe", "audio"):
        config = teacher_config(route, env)
        assert (
            parser.parse_override("+distillation.teacher_models." + route + "=" + hydra_literal(config)).value()
            == config
        )
        assert config["backend"] == "external"
        assert config["key"] == route
        assert config["external"]["max_concurrent_requests"] == (2 if route == "audio" else 4)
