"""Migrate real torchdata sampler state without changing its continuation or source files."""

import copy
import json
import pytest
import torch
import yaml
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from scripts.audio_data.migrate_opd_sampler_checkpoint import migrate_checkpoint
from verl.utils.dataset.audio_opd_sampler import AudioOPDQuotaSampler, dataset_content_fingerprint

SCHEDULE = [
    {"steps": 2, "quotas": {"a": 3, "b": 1}},
    {"steps": 2, "quotas": {"a": 1, "b": 3}},
]


class Pool:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return {"index": index}


def _archive(dataframe, path):
    dataframe.map(lambda row: row, cache_file_name=str(path))
    return Dataset.from_file(str(path))


@pytest.fixture
def original(tmp_path):
    archive = tmp_path / "original.arrow"
    dataframe = _archive(
        Dataset.from_dict({"opd_source": ["a", "b"] * 10, "prompt": [f"prompt {i}" for i in range(20)]}),
        archive,
    )
    pool = Pool(dataframe)
    sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 4, 42)
    sampler.set_step(2)
    loader = StatefulDataLoader(pool, batch_size=1, sampler=sampler, num_workers=0)
    stream = iter(loader)
    for _ in range(5):
        next(stream)
    state = copy.deepcopy(loader.state_dict())
    saved_sampler = state["_sampler_iter_state"]["sampler_state"]
    saved_sampler["version"] = 2
    saved_sampler["identity"] = sampler.legacy_identity
    for field in ("dataset_fingerprint", "schedule", "batch_size"):
        saved_sampler.pop(field, None)
    checkpoint = tmp_path / "original" / "global_step_8"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "weights.bin").write_bytes(b"original actor")
    (checkpoint / "dapo_pending_prompts.pt").write_bytes(b"original queue")
    torch.save(state, checkpoint / "data.pt")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"data": {"value": {"opd_source_mix": SCHEDULE, "train_batch_size": 4}}}))
    return {
        "checkpoint": checkpoint,
        "archive": archive,
        "dataframe": dataframe,
        "config": config,
        "output": tmp_path / "resume" / "global_step_8",
        "state": state,
        "sampler": sampler,
        "stream": stream,
    }


@pytest.mark.parametrize("config_style", ["wandb", "trainer", "data"])
def test_migration_preserves_loader_continuation_and_originals(original, config_style):
    if config_style != "wandb":
        data = {"opd_source_mix": SCHEDULE, "train_batch_size": 4}
        original["config"].write_text(yaml.safe_dump({"data": data} if config_style == "trainer" else data))
    before = {path: path.read_bytes() for path in original["checkpoint"].rglob("*") if path.is_file()}
    archive_before = original["archive"].read_bytes()
    provenance = migrate_checkpoint(original["checkpoint"], original["archive"], original["config"], original["output"])
    migrated = torch.load(original["output"] / "data.pt", weights_only=False)
    new_sampler_state = migrated["_sampler_iter_state"]["sampler_state"]
    assert new_sampler_state["version"] == 3
    assert new_sampler_state["identity"] == original["sampler"].identity
    restored_v2_fields = copy.deepcopy(migrated)
    original_sampler_state = original["state"]["_sampler_iter_state"]["sampler_state"]
    restored_v2_fields["_sampler_iter_state"]["sampler_state"] = original_sampler_state
    assert restored_v2_fields == original["state"]
    for key, value in original["state"]["_sampler_iter_state"]["sampler_state"].items():
        if key not in {"version", "identity"}:
            assert new_sampler_state[key] == value

    # A rebuilt dataframe has a different HF fingerprint, as happens across
    # processes/machines, and still resumes when its ordered content is identical.
    rebuilt = Dataset.from_dict(original["dataframe"].to_dict())
    assert rebuilt._fingerprint != original["dataframe"]._fingerprint
    pool = Pool(rebuilt)
    sampler = AudioOPDQuotaSampler(pool, SCHEDULE, 4, 999)
    loader = StatefulDataLoader(pool, batch_size=1, sampler=sampler, num_workers=0)
    loader.load_state_dict(migrated)
    resumed = iter(loader)
    assert [next(resumed)["index"].item() for _ in range(10)] == [
        next(original["stream"])["index"].item() for _ in range(10)
    ]
    sampler.set_step(3)
    original["sampler"].set_step(3)
    assert [next(resumed)["index"].item() for _ in range(4)] == [
        next(original["stream"])["index"].item() for _ in range(4)
    ]
    assert {path: path.read_bytes() for path in before} == before
    assert original["archive"].read_bytes() == archive_before
    assert {path: path.read_bytes() for path in original["checkpoint"].rglob("*") if path.is_file()} == before
    assert (original["output"] / "actor").is_symlink()
    assert (original["output"] / "actor").resolve() == (original["checkpoint"] / "actor").resolve()
    assert (original["output"] / "dapo_pending_prompts.pt").is_symlink()
    assert not (original["output"] / "data.pt").is_symlink()
    assert json.loads((original["output"] / "migration.json").read_text()) == provenance
    assert provenance["old_identity"] == original["sampler"].legacy_identity
    assert provenance["dataset_content_fingerprint"] == dataset_content_fingerprint(rebuilt)
    assert provenance["quota_source_order"] == [["a", "b"], ["a", "b"]]


def test_migrated_state_rejects_runtime_quota_key_reordering(original):
    migrate_checkpoint(original["checkpoint"], original["archive"], original["config"], original["output"])
    migrated = torch.load(original["output"] / "data.pt", weights_only=False)
    reordered = copy.deepcopy(SCHEDULE)
    for phase in reordered:
        phase["quotas"] = dict(reversed(list(phase["quotas"].items())))
    sampler = AudioOPDQuotaSampler(Pool(original["dataframe"]), reordered, 4)
    with pytest.raises(ValueError, match="different data"):
        sampler.load_state_dict(migrated["_sampler_iter_state"]["sampler_state"])


@pytest.mark.parametrize("change", ["prompt", "row_order", "schedule", "batch_size"])
def test_migration_rejects_archive_or_curriculum_changes_before_writing(original, tmp_path, change):
    archive = original["archive"]
    data = {"opd_source_mix": copy.deepcopy(SCHEDULE), "train_batch_size": 4}
    if change == "prompt":
        changed = original["dataframe"].map(lambda row: {"prompt": row["prompt"] + " changed"})
        archive = tmp_path / "changed.arrow"
        _archive(changed, archive)
    elif change == "row_order":
        # Swapping rows within one source keeps source-group indices identical.
        order = list(range(20))
        order[0], order[2] = order[2], order[0]
        archive = tmp_path / "reordered.arrow"
        _archive(original["dataframe"].select(order), archive)
    elif change == "schedule":
        data["opd_source_mix"][0]["steps"] += 1
    else:
        data["train_batch_size"] = 2
        for phase in data["opd_source_mix"]:
            phase["quotas"] = {"a": 1, "b": 1}
    original["config"].write_text(yaml.safe_dump({"data": data}))
    with pytest.raises(ValueError, match="different data"):
        migrate_checkpoint(original["checkpoint"], archive, original["config"], original["output"])
    assert not original["output"].exists()


@pytest.mark.parametrize("collision", ["directory", "dangling_symlink"])
def test_migration_never_overwrites_output(original, collision):
    original["output"].parent.mkdir()
    if collision == "directory":
        original["output"].mkdir()
        marker = original["output"] / "keep"
        marker.write_text("keep")
    else:
        original["output"].symlink_to(original["output"].parent / "missing")
    with pytest.raises(FileExistsError, match="already exists"):
        migrate_checkpoint(original["checkpoint"], original["archive"], original["config"], original["output"])
    if collision == "directory":
        assert marker.read_text() == "keep"
    else:
        assert original["output"].is_symlink()


@pytest.mark.parametrize("invalid_output", ["wrong_step", "inside_original"])
def test_migration_requires_a_separate_checkpoint_with_the_same_step(original, invalid_output):
    output = (
        original["output"].with_name("global_step_9")
        if invalid_output == "wrong_step"
        else original["checkpoint"] / "nested" / original["checkpoint"].name
    )
    with pytest.raises(ValueError, match="same global_step_N|outside the original"):
        migrate_checkpoint(original["checkpoint"], original["archive"], original["config"], output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["missing_sampler", "multiple_samplers", "already_v3"])
def test_migration_rejects_ambiguous_or_already_migrated_state(original, problem):
    state = copy.deepcopy(original["state"])
    if problem == "missing_sampler":
        del state["_sampler_iter_state"]["sampler_state"]
    elif problem == "multiple_samplers":
        state["another"] = {"sampler_state": copy.deepcopy(state["_sampler_iter_state"]["sampler_state"])}
    else:
        state["_sampler_iter_state"]["sampler_state"]["version"] = 3
    torch.save(state, original["checkpoint"] / "data.pt")
    with pytest.raises(ValueError, match="exactly one nested|version 2"):
        migrate_checkpoint(original["checkpoint"], original["archive"], original["config"], original["output"])
    assert not original["output"].exists()
