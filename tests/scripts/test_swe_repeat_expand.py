"""Frozen suite integrity, exact evidence migration and task isolation."""
import gzip
import json

import pytest

from recipe.swe_agent import repetition_reward as policy
from scripts.swe_repeat_expand import build, canonical_task_key
from scripts.swe_repeat_regression import file_digest, partition, replay_events, write_json


@pytest.fixture
def inputs(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    task = "django__django-15957"
    case = {"id": "old", "task": task, "task_key": "django/django\x1f" + task,
            "partition": "old-split", "public_events": [], "public_context": "old context",
            "raw": {"raw_score": 1}}
    with gzip.open(existing / "fixtures.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(case) + "\n")
    write_json(existing / "labels.json", {"policy_version": policy.POLICY_VERSION,
        "cases": [{"id": "old", "expected": "clean", "reason": "Distinct investigation."}]})
    write_json(existing / "manifest.json", {
        "fixtures_sha256": file_digest(existing / "fixtures.jsonl.gz"),
        "labels_sha256": file_digest(existing / "labels.json")})
    capture = tmp_path / "capture.json.gz"
    artifact = {"version": "repeat-public-events-v2", "original_public_context": "captured context",
                "events": [{"id": 1, "name": "Read", "arguments": {"file_path": "a.py"},
                            **policy.capture_observation("exact original output")}],
                "messages": [{"role": "user", "content": "old prompt preserved by hash"}]}
    with gzip.open(capture, "wt") as stream:
        json.dump(artifact, stream)
    reviews = tmp_path / "reviews.json"
    write_json(reviews, {"cases": [{"id": "new", "task": task, "source": str(capture),
        "sha256": file_digest(capture), "expected": "uncertain", "reason": "Limited evidence.", "raw_score": 0}]})
    return existing, reviews, tmp_path / "frozen", artifact


def test_exact_capture_migration_and_canonical_task_partition(inputs):
    existing, reviews, output, artifact = inputs
    manifest = build(existing, reviews, output)
    with gzip.open(output / "fixtures.jsonl.gz", "rt") as stream:
        cases = {case["id"]: case for case in map(json.loads, stream)}
    old, new = cases["old"], cases["new"]
    assert old["task_key"] == new["task_key"] == "django__django-15957"
    assert old["partition"] == new["partition"] == partition(new["task_key"])
    events, context, source = replay_events(new, policy)
    assert events == artifact["events"] and context == artifact["original_public_context"]
    assert source == "captured_events"
    assert new["capture_serializer"] == "repeat-public-events-v2"
    assert new["capture_messages_sha256"] == policy.digest(artifact["messages"])
    assert manifest["fixtures_sha256"] == file_digest(output / "fixtures.jsonl.gz")
    assert manifest["labels_sha256"] == file_digest(output / "labels.json")
    labels = json.loads((output / "labels.json").read_text())["cases"]
    assert next(row for row in labels if row["id"] == "old")["expected"] == "clean"
    assert len(cases) == len(labels) == len({row["id"] for row in labels})
    with pytest.raises(ValueError, match="overwrite"):
        build(existing, reviews, output)


@pytest.mark.parametrize("target", ["fixtures", "labels", "capture"])
def test_hash_mismatch_refuses_to_freeze(inputs, target):
    existing, reviews, output, _ = inputs
    path = {"fixtures": existing / "fixtures.jsonl.gz", "labels": existing / "labels.json",
            "capture": reviews.parent / "capture.json.gz"}[target]
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="integrity|hash"):
        build(existing, reviews, output)
    assert not output.exists()


@pytest.mark.parametrize("case,expected", [
    ({"task_key": "repo\x1finstance"}, "instance"),
    ({"task_key": "repo\x1finstance", "task": "canonical"}, "canonical"),
    ({"task_key": "synthetic:loop"}, "synthetic:loop"),
])
def test_canonical_task_identity(case, expected):
    assert canonical_task_key(case) == expected
