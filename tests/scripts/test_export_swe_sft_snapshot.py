import copy
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import export_swe_sft_snapshot as snapshot
from scripts import swe_sft_quality as quality
from tests.scripts.test_swe_sft_quality import fixture_row, good


def fixture_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "rollouts" / "source"
    source.mkdir(parents=True)
    rows = [fixture_row("accepted"), fixture_row("pending")]
    rows[1]["input"] = rows[1]["input"].replace("Fix a.py.", "Fix b.py.")
    (source / "1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    reference = tmp_path / "reference.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "index": index,
                    "extra_info": {"benchmark": "swe_rebench_v2", "task_id": task},
                    "payload": f"row{index}",
                }
                for index, task in enumerate(["accepted", "accepted", "pending", "rl-only"])
            ]
        ),
        reference,
    )
    args = snapshot.build_parser().parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(tmp_path / "snapshot"),
            "--rl-reference-parquet",
            str(reference),
            "--write-filtered-rl",
        ]
    )
    selected = snapshot.selection_args(args)
    selected.rollout_root = source.parent
    selected.runs = [source.name]
    db, _, _ = quality.prepare_pool(selected)
    digest, packed, provenance = db.execute(
        "SELECT hash,raw,provenance FROM candidates WHERE task=?", (json.dumps(["swe_rebench_v2", "accepted"]),)
    ).fetchone()
    db.close()
    provenance = json.loads(provenance)
    events, segments, evidence_source = quality.load_evidence(json.loads(gzip.decompress(packed)))
    result = {
        "accepted": True,
        "verdict": "good",
        "stage": "model",
        "error": 0,
        "good_votes": 8,
        "vote_count": 8,
        "vote_quorum": 8,
        "votes": [copy.deepcopy(good()) for _ in range(8)],
        "reviewer_called": True,
        "reviewer": good(),
    }
    config = quality.dataclasses.asdict(quality.judge_config(selected))
    artifact = {
        "transcript_hash": digest,
        "source": provenance,
        "public_task": rows[0]["input"],
        "policy_source_sha256": quality.SOURCE_SHA256,
        "result": result,
        "config": config,
        "events": events,
        "actor_segments": segments,
        "provenance": {"policy_source_sha256": quality.repeat.POLICY_SOURCE_SHA256, "served_root": "fixture-only"},
    }
    artifacts = run_dir / "quality_artifacts"
    artifacts.mkdir()
    artifact_path = artifacts / f"{digest}.json.gz"
    artifact_path.write_bytes(gzip.compress(json.dumps(artifact).encode()))
    record = {
        **provenance,
        "transcript_hash": digest,
        "task_key": json.dumps(["swe_rebench_v2", "accepted"]),
        "task_id": "accepted",
        "accepted": True,
        "verdict": "good",
        "stage": "model",
        "good_votes": 8,
        "vote_count": 8,
        "reviewer_called": True,
        "reviewer_verdict": "good",
        "unresolved_error_calls": 0,
        "artifact_path": str(artifact_path),
        "artifact_sha256": snapshot.sha256(artifact_path),
        "evidence_source": evidence_source,
    }
    (run_dir / "quality_decisions.jsonl").write_text(json.dumps(record) + '\n{"accepted":')
    (run_dir / "run_status.json").write_text('{"status":"filter_running","unaudited_override":true}')
    return args, artifact_path, record


def test_live_snapshot_reserves_pending_tasks_preserves_schema_and_run(tmp_path, monkeypatch):
    args, _, _ = fixture_run(tmp_path)
    originals = {p: snapshot.sha256(p) for p in args.run_dir.rglob("*") if p.is_file()}
    reference_hash = snapshot.sha256(args.rl_reference_parquet)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Snapshot must never call a model")

    monkeypatch.setattr(quality.SFTJudgeClient, "preflight", forbidden)
    monkeypatch.setattr(quality.SFTJudgeClient, "judge_consensus", forbidden)
    report = snapshot.export_snapshot(args)
    assert report["accepted_unique_tasks"] == report["dataset_counts"]["core_train"] == 1
    assert report["cutoff"]["ignored_partial_bytes"] == len('{"accepted":')
    assert not report["independent_audit_verified"]
    assert report["rl_exclusion"]["matched_reference_rows"] == 3
    assert report["rl_exclusion"]["remaining_task_overlap"] == 0
    filtered = pq.read_table(args.output_dir / "rl_train_filtered.parquet")
    assert filtered.schema == pq.read_table(args.rl_reference_parquet).schema
    assert filtered.to_pylist()[0]["payload"] == "row3"
    assert [
        json.loads(line)["task_id"] for line in (args.output_dir / "sft_prompt_ids.jsonl").read_text().splitlines()
    ] == ["accepted"]
    assert {
        json.loads(line)["task_id"]
        for line in (args.output_dir / "rl_reserved_prompt_ids.jsonl").read_text().splitlines()
    } == {"accepted", "pending"}
    assert all(snapshot.sha256(path) == digest for path, digest in originals.items())
    assert snapshot.sha256(args.rl_reference_parquet) == reference_hash
    assert all(snapshot.sha256(args.output_dir / path) == digest for path, digest in report["files"].items())
    with pytest.raises(FileExistsError, match="already exists"):
        snapshot.export_snapshot(args)


def test_snapshot_scope_excludes_only_accepted_and_can_skip_rl_copy(tmp_path):
    args, _, _ = fixture_run(tmp_path)
    args.rl_exclusion_scope = "snapshot"
    args.write_filtered_rl = False
    report = snapshot.export_snapshot(args)
    assert report["rl_exclusion"]["matched_reference_rows"] == 2
    assert report["rl_exclusion"]["remaining_reference_rows"] == 2
    assert not (args.output_dir / "rl_train_filtered.parquet").exists()


def test_preserves_historical_reward_version_from_live_process(tmp_path):
    args, artifact_path, record = fixture_run(tmp_path)
    artifact = json.loads(gzip.decompress(artifact_path.read_bytes()))
    historical_hash = "a" * 64
    artifact["provenance"]["policy_source_sha256"] = historical_hash
    artifact_path.write_bytes(gzip.compress(json.dumps(artifact).encode()))
    record["artifact_sha256"] = snapshot.sha256(artifact_path)
    (args.run_dir / "quality_decisions.jsonl").write_text(json.dumps(record) + "\n")
    report = snapshot.export_snapshot(args)
    assert report["recorded_reward_source_differs_from_exporter"]
    assert report["judge_provenances"][0]["policy_source_sha256"] == historical_hash
    assert report["exporter_reward_source_sha256"] == quality.repeat.POLICY_SOURCE_SHA256


@pytest.mark.parametrize(
    "damage", ["artifact_hash", "guard", "reviewer", "identity", "fingerprint", "duplicate", "malformed"]
)
def test_invalid_checkpoint_never_publishes(tmp_path, damage):
    args, artifact_path, record = fixture_run(tmp_path)
    if damage == "fingerprint":
        args.selection_args_json = tmp_path / "selection.json"
        args.selection_args_json.write_text('{"max_response_tokens":123}')
    elif damage in {"artifact_hash", "guard", "reviewer", "identity"}:
        artifact = json.loads(gzip.decompress(artifact_path.read_bytes()))
        if damage == "guard":
            artifact["result"]["votes"][0]["event_ids"] = [999]
        elif damage == "reviewer":
            artifact["result"]["reviewer"]["checks"]["productive_tools"] = False
        else:
            artifact["public_task"] = "different task"
        artifact_path.write_bytes(gzip.compress(json.dumps(artifact).encode()))
        if damage != "artifact_hash":
            record["artifact_sha256"] = snapshot.sha256(artifact_path)
    log = args.run_dir / "quality_decisions.jsonl"
    log.write_text(json.dumps(record) + "\n")
    if damage == "duplicate":
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
    if damage == "malformed":
        with log.open("a") as f:
            f.write("not JSON\n")
    with pytest.raises(ValueError):
        snapshot.export_snapshot(args)
    assert not args.output_dir.exists()
    assert not list(tmp_path.glob(".snapshot.*"))


def test_capture_ignores_new_decisions_after_observed_cutoff(tmp_path, monkeypatch):
    source = tmp_path / "decisions.jsonl"
    source.write_text('{"accepted":true}\n')
    original_fstat = snapshot.os.fstat
    first = True

    def append_after_stat(fd):
        nonlocal first
        status = original_fstat(fd)
        if first:
            first = False
            with source.open("a") as f:
                f.write('{"accepted":false}\n')
        return status

    monkeypatch.setattr(snapshot.os, "fstat", append_after_stat)
    records, report = snapshot.capture_decisions(source, tmp_path / "captured.jsonl")
    assert records == [{"accepted": True}]
    assert report["captured_bytes"] == len('{"accepted":true}\n')


def test_empty_acceptance_and_source_output_are_rejected(tmp_path):
    args, _, _ = fixture_run(tmp_path)
    (args.run_dir / "quality_decisions.jsonl").write_text('{"accepted":false,"verdict":"bad"}\n')
    with pytest.raises(ValueError, match="No accepted"):
        snapshot.export_snapshot(args)
    args.output_dir = args.run_dir / "snapshot"
    with pytest.raises(ValueError, match="separate"):
        snapshot.export_snapshot(args)
