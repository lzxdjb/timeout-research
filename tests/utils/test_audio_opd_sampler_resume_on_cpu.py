"""Resume OPD sampling across HF provenance changes without accepting changed rows."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, concatenate_datasets
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.utils.dataset.audio_opd_sampler import AudioOPDQuotaSampler, dataset_content_fingerprint
from verl.utils.dataset.rl_dataset import RLHFDataset


SCHEDULE = [
    {"steps": 3, "quotas": {"swe": 3, "audio": 1}},
    {"steps": 3, "quotas": {"swe": 1, "audio": 3}},
]


class TinyTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt and tokenize
        return list(range(sum(len(message["content"].split()) for message in messages)))


class RowPool:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return {"index": index, "sample_id": self.dataframe[index]["sample_id"]}


def _rows():
    return [
        {
            "sample_id": f"{source}-{index}",
            "opd_source": source,
            "teacher_route": "swe" if source == "swe" else "audio",
            "prompt": [{"role": "user", "content": f"sample {source} {index}"}],
            "audios": [{"audio": f"/audio/{index}.wav"}] if source == "audio" else None,
            "reward_model": {"ground_truth": f"answer-{index}"},
            "extra_info": {"task_id": f"task-{source}-{index}"},
        }
        for source in ("swe", "audio")
        for index in range(12)
    ]


def _filter_with_config(dataframe, config):
    # Exercise the real filter closure, including RLHFDataset's serialization
    # and OmegaConf's parent links, without downloading a model or dataset.
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.config = config
    dataset.serialize_dataset = False
    dataset.filter_overlong_prompts = config.filter_overlong_prompts
    dataset.tokenizer = TinyTokenizer()
    dataset.processor = None
    dataset.prompt_key = config.prompt_key
    dataset.max_prompt_length = config.max_prompt_length
    dataset.num_workers = config.filter_overlong_prompts_workers
    dataset.apply_chat_template_kwargs = config.apply_chat_template_kwargs
    dataset.tool_schemas = None
    return dataset.maybe_filter_out_long_prompts(dataframe)


@pytest.fixture(scope="module")
def filtered_dataframes():
    original_config = OmegaConf.create(
        {
            "data": {
                "prompt_key": "prompt",
                "max_prompt_length": 4,
                "filter_overlong_prompts": True,
                "filter_overlong_prompts_workers": 1,
                "apply_chat_template_kwargs": {},
            },
            "trainer": {"resume_mode": "disable", "resume_from_path": None},
        }
    )
    resume_config = copy.deepcopy(original_config)
    resume_config.trainer.resume_mode = "resume_path"
    resume_config.trainer.resume_from_path = "/checkpoints/global_step_2"
    assert original_config.data == resume_config.data

    rows = _rows()
    overlong = copy.deepcopy(rows[0])
    overlong["sample_id"] = "filtered-out"
    overlong["prompt"][0]["content"] = "one two three four five"
    dataframe = Dataset.from_list([*rows, overlong])
    original = _filter_with_config(dataframe, original_config.data)
    resumed = _filter_with_config(dataframe, resume_config.data)

    assert original.to_list() == resumed.to_list() == rows
    # The production bug: the real filter hashes the unrelated trainer config
    # through self.config's parent, even though all retained rows are identical.
    assert original._fingerprint != resumed._fingerprint
    return original, resumed


def _loader(dataframe, seed=42):
    pool = RowPool(dataframe)
    sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 4, seed)
    return sampler, StatefulDataLoader(pool, batch_size=1, sampler=sampler, num_workers=0)


def _draw_indices(stream, count):
    return [next(stream)["index"].item() for _ in range(count)]


def _legacy_state(sampler):
    # Reproduce the version-2 schema and identity independently of the new
    # state_dict implementation, so compatibility cannot pass by construction.
    identity = {
        "fingerprint": sampler.dataset.dataframe._fingerprint,
        "groups": dict(sampler.groups),
        "schedule": sampler.schedule,
        "batch_size": sampler.batch_size,
    }
    state = sampler.state_dict()
    state["version"] = 2
    state["identity"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for key in ("dataset_fingerprint", "schedule", "batch_size"):
        state.pop(key, None)
    return state


def test_filtered_resume_config_changes_preserve_exact_loader_sequence(filtered_dataframes):
    original, resumed = filtered_dataframes
    sampler, loader = _loader(original)
    stream = iter(loader)
    _draw_indices(stream, 5)
    assert sampler.pending  # Save in the middle of a quota round.
    saved = copy.deepcopy(loader.state_dict())
    assert sampler.state_dict()["version"] == 3
    expected = _draw_indices(stream, 9)

    resumed_sampler, resumed_loader = _loader(resumed, seed=999)
    assert resumed_sampler.identity == sampler.identity
    assert resumed_sampler.legacy_identity != sampler.legacy_identity
    resumed_loader.load_state_dict(saved)
    resumed_stream = iter(resumed_loader)
    assert _draw_indices(resumed_stream, 9) == expected

    sampler.set_step(4)
    resumed_sampler.set_step(4)
    assert _draw_indices(resumed_stream, 8) == _draw_indices(stream, 8)
    assert resumed_sampler.state_dict() == sampler.state_dict()


def test_content_fingerprint_ignores_format_columns_and_arrow_chunk_boundaries(filtered_dataframes):
    dataframe, _ = filtered_dataframes
    rechunked = concatenate_datasets([dataframe.select(range(7)), dataframe.select(range(7, len(dataframe)))])
    reformatted = rechunked.select_columns(list(reversed(dataframe.column_names))).with_format("numpy")
    assert reformatted._fingerprint != dataframe._fingerprint
    assert dataset_content_fingerprint(reformatted) == dataset_content_fingerprint(dataframe)


@pytest.mark.parametrize("change", ["prompt", "reference", "audio_path", "routing", "metadata", "same_source_reorder"])
def test_actual_row_changes_reject_resume_even_with_same_hf_fingerprint(filtered_dataframes, change):
    original, _ = filtered_dataframes
    rows = original.to_list()
    if change == "prompt":
        rows[0]["prompt"][0]["content"] = "different sample prompt"
    elif change == "reference":
        rows[0]["reward_model"]["ground_truth"] = "different reference"
    elif change == "audio_path":
        rows[12]["audios"][0]["audio"] = "/audio/replaced.wav"
    elif change == "routing":
        rows[0]["teacher_route"] = "another-teacher"
    elif change == "metadata":
        rows[0]["extra_info"]["task_id"] = "different-task"
    else:
        rows[0], rows[1] = rows[1], rows[0]
    changed = Dataset.from_list(rows)
    changed._fingerprint = original._fingerprint

    sampler, _ = _loader(original)
    changed_sampler, _ = _loader(changed)
    assert changed_sampler.groups == sampler.groups
    assert changed_sampler.legacy_identity == sampler.legacy_identity
    before = changed_sampler.state_dict()
    with pytest.raises(ValueError, match="different data, batch size or curriculum"):
        changed_sampler.load_state_dict(sampler.state_dict())
    assert changed_sampler.state_dict() == before


@pytest.mark.parametrize("change", ["phase_duration", "source_quotas", "source_order", "batch_size"])
def test_changed_curriculum_or_batch_size_rejects_resume(filtered_dataframes, change):
    original, _ = filtered_dataframes
    sampler, _ = _loader(original)
    schedule = copy.deepcopy(SCHEDULE)
    batch_size = 4
    if change == "phase_duration":
        schedule[0]["steps"] = 2
    elif change == "source_quotas":
        schedule[0]["quotas"] = {"swe": 2, "audio": 2}
    elif change == "source_order":
        schedule[0]["quotas"] = dict(reversed(list(schedule[0]["quotas"].items())))
    else:
        batch_size = 8
        for phase in schedule:
            phase["quotas"] = {source: count * 2 for source, count in phase["quotas"].items()}
    changed_sampler = AudioOPDQuotaSampler(RowPool(original), schedule, batch_size)
    with pytest.raises(ValueError, match="different data, batch size or curriculum"):
        changed_sampler.load_state_dict(sampler.state_dict())


def test_version_2_checkpoint_with_unchanged_legacy_identity_resumes_exactly(filtered_dataframes):
    original, _ = filtered_dataframes
    sampler, _ = _loader(original)
    stream = iter(sampler)
    for _ in range(5):
        next(stream)
    saved = _legacy_state(sampler)
    expected = [next(stream) for _ in range(17)]

    resumed_sampler, _ = _loader(original, seed=999)
    assert resumed_sampler.legacy_identity == saved["identity"]
    resumed_sampler.load_state_dict(saved)
    resumed_stream = iter(resumed_sampler)
    assert [next(resumed_stream) for _ in range(17)] == expected
    assert resumed_sampler.state_dict()["version"] == 3
    assert resumed_sampler.state_dict() == sampler.state_dict()


def test_version_2_hf_fingerprint_change_requires_explicit_migration(filtered_dataframes):
    original, resumed = filtered_dataframes
    sampler, _ = _loader(original)
    resumed_sampler, _ = _loader(resumed)
    assert resumed_sampler.identity == sampler.identity
    before = resumed_sampler.state_dict()
    with pytest.raises(ValueError, match="Version-2 checkpoints also depend on the HF cache fingerprint"):
        resumed_sampler.load_state_dict(_legacy_state(sampler))
    assert resumed_sampler.state_dict() == before


@pytest.mark.parametrize("changed_content", [False, True])
def test_trainer_checkpoint_validates_sampler_before_fetch_without_consuming_rows(
    filtered_dataframes, tmp_path, changed_content
):
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    original, resumed = filtered_dataframes
    sampler, loader = _loader(original)
    stream = iter(loader)
    _draw_indices(stream, 5)
    saved_sampler_state = sampler.state_dict()
    checkpoint = tmp_path / "global_step_2"
    checkpoint.mkdir()
    torch.save(loader.state_dict(), checkpoint / "data.pt")
    expected = _draw_indices(stream, 9)

    if changed_content:
        rows = resumed.to_list()
        rows[0]["prompt"][0]["content"] = "changed after checkpoint"
        resumed = Dataset.from_list(rows)
    resumed_sampler, resumed_loader = _loader(resumed, seed=999)
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "data": {"opd_source_mix": SCHEDULE},
                "trainer": {
                    "resume_mode": "resume_path",
                    "resume_from_path": str(checkpoint),
                    "del_local_ckpt_after_load": False,
                },
            }
        ),
        actor_rollout_wg=SimpleNamespace(load_checkpoint=lambda **kwargs: None),
        _load_timeout_predictor=lambda path: None,
        use_critic=False,
        trainer_mode="sync",
        train_dataloader=resumed_loader,
        train_dataloader_it=None,
    )
    if changed_content:
        with pytest.raises(ValueError, match="different data, batch size or curriculum"):
            PPOTrainer._load_checkpoint(trainer)
        assert not resumed_sampler.draw_counts
    else:
        PPOTrainer._load_checkpoint(trainer)
        assert trainer.global_steps == 2
        assert trainer.train_dataloader_it is not None
        assert resumed_sampler.state_dict() == saved_sampler_state
        assert _draw_indices(trainer.train_dataloader_it, 9) == expected
