import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import mix_swe_sft_data as mixture


def row(task_id, *, teacher=False, length=100):
    return {
        "benchmark": "swe_rebench_v2",
        "task_id": task_id,
        "task_key": task_id,
        "raw_score": 1.0,
        "response_tokens": length,
        "messages": [
            {"role": "user", "content": "Fix the implementation."},
            {
                "role": "assistant",
                "content": "The implementation is fixed.",
                **({"reasoning_content": "Check the boundary condition."} if teacher else {}),
            },
        ],
        "transcript_hash": f"{'teacher' if teacher else 'existing'}-{task_id}",
    }


def fixture_args(tmp_path, *, oversized=False):
    existing = tmp_path / "existing.parquet"
    teacher = tmp_path / "teacher.parquet"
    existing_rows = [row(f"existing-{i}") for i in range(40)] + [row("shared")]
    teacher_rows = [row(f"teacher-{i}", teacher=True) for i in range(20)] + [row("shared", teacher=True)]
    if oversized:
        teacher_rows.append(row("oversized", teacher=True, length=2000))
    pq.write_table(pa.Table.from_pylist(existing_rows), existing)
    pq.write_table(pa.Table.from_pylist(teacher_rows), teacher)
    held_out = tmp_path / "held-out.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"extra_info": {"benchmark": "swe_rebench_v2", "task_id": "existing-0"}}]), held_out
    )
    return mixture.build_parser().parse_args(
        [
            "--existing-parquet",
            str(existing),
            "--teacher-parquet",
            str(teacher),
            "--exclude-task-parquet",
            str(held_out),
            "--output-dir",
            str(tmp_path / "mix"),
            "--tokenizer-path",
            str(tmp_path / "student"),
            "--max-length",
            "1000",
        ]
    )


def test_mixture_preserves_tasks_reasoning_and_holds_out_validation(tmp_path):
    args = fixture_args(tmp_path)
    original_hashes = {path: mixture.sha256(path) for path in args.existing_parquet + args.teacher_parquet}
    report = mixture.mix(args, token_counter=lambda r: r["response_tokens"])
    train = pq.read_table(args.output_dir / "train.parquet").to_pylist()
    val = pq.read_table(args.output_dir / "val.parquet").to_pylist()
    train_tasks = {mixture.task_key(r) for r in train}
    val_tasks = {mixture.task_key(r) for r in val}
    assert train_tasks.isdisjoint(val_tasks)
    assert len(val_tasks) == len(val)
    assert len(train_tasks | val_tasks) == 60
    assert ("swe_rebench_v2", "existing-0") not in train_tasks | val_tasks
    shared = [r for r in train + val if r["task_id"] == "shared"]
    assert shared and all(r["sft_origin"] == "teacher" for r in shared)
    assert shared[0]["messages"][1]["reasoning_content"] == "Check the boundary condition."
    assert all(r["student_sequence_tokens"] == 100 for r in train + val)
    assert report["teacher_fraction_actual"] == report["train_rows"]["teacher"] / len(train)
    assert len(train) == len(train_tasks)
    assert report["sampling_mode"] == "downsample"
    assert report["repeated_training_rows"] == 0
    assert sum(report["train_unique_tasks"].values()) == len(train_tasks)
    assert all(mixture.sha256(path) == digest for path, digest in original_hashes.items())
    assert all(mixture.sha256(args.output_dir / path) == digest for path, digest in report["files"].items())
    with pytest.raises(FileExistsError, match="already exists"):
        mixture.mix(args, token_counter=lambda r: 100)


def test_student_context_limit_is_applied_before_mixing(tmp_path):
    args = fixture_args(tmp_path, oversized=True)
    report = mixture.mix(args, token_counter=lambda r: r["response_tokens"])
    assert report["excluded_counts"]["student_context_limit"] == 1
    excluded = [json.loads(line) for line in (args.output_dir / "excluded_tasks.jsonl").read_text().splitlines()]
    assert {
        "benchmark": "swe_rebench_v2",
        "task_id": "oversized",
        "reason": "student_context_limit",
        "tokens": 2000,
    } in excluded
    manifest = [json.loads(line) for line in (args.output_dir / "task_manifest.jsonl").read_text().splitlines()]
    assert "oversized" not in {r["task_id"] for r in manifest}


def test_oversized_teacher_keeps_usable_existing_example_of_same_task(tmp_path):
    args = fixture_args(tmp_path)
    rows = pq.read_table(args.teacher_parquet[0]).to_pylist()
    next(r for r in rows if r["task_id"] == "shared")["response_tokens"] = 2000
    pq.write_table(pa.Table.from_pylist(rows), args.teacher_parquet[0])
    mixture.mix(args, token_counter=lambda r: r["response_tokens"])
    selected = (
        pq.read_table(args.output_dir / "train.parquet").to_pylist()
        + pq.read_table(args.output_dir / "val.parquet").to_pylist()
    )
    assert {r["sft_origin"] for r in selected if r["task_id"] == "shared"} == {"existing"}


@pytest.mark.parametrize("fraction,origin", [(0, "existing"), (1, "teacher")])
def test_single_source_ablation_does_not_require_other_snapshot(tmp_path, fraction, origin):
    args = fixture_args(tmp_path)
    args.teacher_fraction = fraction
    args.val_fraction = 0.5
    if origin == "teacher":
        args.existing_parquet = []
    else:
        args.teacher_parquet = []
    report = mixture.mix(args, token_counter=lambda r: r["response_tokens"])
    train = pq.read_table(args.output_dir / "train.parquet").to_pylist()
    val = pq.read_table(args.output_dir / "val.parquet").to_pylist()
    assert {r["sft_origin"] for r in train + val} == {origin}
    assert report["teacher_fraction_actual"] == fraction
    assert report["train_rows"][origin] == report["train_unique_tasks"][origin]
    assert {mixture.task_key(r) for r in train}.isdisjoint(mixture.task_key(r) for r in val)


def test_missing_requested_source_fails_before_publishing(tmp_path):
    args = fixture_args(tmp_path)
    args.existing_parquet = []
    with pytest.raises(ValueError, match="existing-parquet is required"):
        mixture.mix(args, token_counter=lambda r: 100)
    assert not args.output_dir.exists()


def test_mixture_is_deterministic_for_fixed_seed(tmp_path):
    args = fixture_args(tmp_path)
    mixture.mix(args, token_counter=lambda r: 100)
    first = pq.read_table(args.output_dir / "train.parquet").to_pylist()
    args.output_dir = tmp_path / "second"
    mixture.mix(args, token_counter=lambda r: 100)
    assert pq.read_table(args.output_dir / "train.parquet").to_pylist() == first


def test_failed_success_check_does_not_publish(tmp_path):
    args = fixture_args(tmp_path)
    rows = pq.read_table(args.teacher_parquet[0]).to_pylist()
    rows[0]["raw_score"] = 0.0
    pq.write_table(pa.Table.from_pylist(rows), args.teacher_parquet[0])
    with pytest.raises(ValueError, match="Non-successful"):
        mixture.mix(args, token_counter=lambda r: 100)
    assert not args.output_dir.exists()


def test_downsampling_keeps_teacher_tasks_without_repetition(tmp_path):
    args = fixture_args(tmp_path)
    args.teacher_fraction = 0.8
    report = mixture.mix(args, token_counter=lambda r: 100)
    train = pq.read_table(args.output_dir / "train.parquet").to_pylist()
    assert len({r["task_id"] for r in train}) == len(train)
    assert report["train_unique_tasks"]["teacher"] == report["eligible_train_unique_tasks"]["teacher"]
    assert report["train_unique_tasks"]["existing"] < report["eligible_train_unique_tasks"]["existing"]
    assert abs(report["teacher_fraction_actual"] - 0.8) < 1 / len(train)
    manifest = [json.loads(line) for line in (args.output_dir / "task_manifest.jsonl").read_text().splitlines()]
    assert {r["task_id"] for r in manifest if r["split"] == "train"} == {r["task_id"] for r in train}


def test_explicit_oversampling_preserves_tasks_and_reports_repeats(tmp_path):
    args = fixture_args(tmp_path)
    args.sampling_mode = "oversample"
    report = mixture.mix(args, token_counter=lambda r: 100)
    assert report["repeated_training_rows"] > 0
    assert report["train_unique_tasks"] == report["eligible_train_unique_tasks"]
    assert abs(report["teacher_fraction_actual"] - 0.3) < 1 / sum(report["train_rows"].values())


def test_held_out_and_deduplication_apply_across_benchmark_aliases(tmp_path):
    args = fixture_args(tmp_path)
    existing = pq.read_table(args.existing_parquet[0]).to_pylist()
    next(r for r in existing if r["task_id"] == "existing-0")["benchmark"] = "validation-benchmark-alias"
    pq.write_table(pa.Table.from_pylist(existing), args.existing_parquet[0])
    teacher = pq.read_table(args.teacher_parquet[0]).to_pylist()
    shared = next(r for r in teacher if r["task_id"] == "shared")
    shared["benchmark"] = "teacher-benchmark-alias"
    teacher.append({**shared, "benchmark": "another-teacher-alias"})
    pq.write_table(pa.Table.from_pylist(teacher), args.teacher_parquet[0])
    report = mixture.mix(args, token_counter=lambda r: 100)
    rows = (
        pq.read_table(args.output_dir / "train.parquet").to_pylist()
        + pq.read_table(args.output_dir / "val.parquet").to_pylist()
    )
    assert "existing-0" not in {r["task_id"] for r in rows}
    assert len([r for r in rows if r["task_id"] == "shared"]) == 1
    assert next(r for r in rows if r["task_id"] == "shared")["sft_origin"] == "teacher"
    assert report["train_val_task_overlap"] == 0


def test_token_counter_reuses_training_turn_rendering_and_sanity_check(monkeypatch, tmp_path):
    import torch
    import transformers

    from verl.utils.dataset import multiturn_sft_dataset
    from verl.utils.tokenizer import chat_template

    calls = []

    class FakeDataset:
        def _process_single_message(self, **kwargs):
            calls.append(kwargs)
            ids = torch.tensor([1, 2, 3])
            return ids, torch.ones_like(ids), torch.ones_like(ids), {}

        def sanity_check(self, ids, messages, tools, thinking):
            assert ids.numel() == 6
            assert messages[0]["role"] == "system"
            assert tools == [{"name": "bash"}]
            assert thinking is True
            calls.append("sanity")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(multiturn_sft_dataset, "MultiTurnSFTDataset", FakeDataset)
    monkeypatch.setattr(chat_template, "extract_system_prompt_and_generation", lambda *a, **kw: ([], []))
    value = row("one", teacher=True)
    value["messages"].insert(0, {"role": "system", "content": "Use the tools."})
    value["tools"] = [{"name": "bash"}]
    assert mixture.make_token_counter(tmp_path, True)(value) == 6
    assert calls[0]["message"] == value["messages"][:2]
    assert calls[0]["tools"] == value["tools"]
    assert calls[1]["message"] == value["messages"][2]
    assert calls[1]["tools"] is None
    assert calls[-1] == "sanity"


def test_token_counter_rejects_training_serialization_mismatch(monkeypatch, tmp_path):
    import torch
    import transformers

    from verl.utils.dataset import multiturn_sft_dataset
    from verl.utils.tokenizer import chat_template

    class FakeDataset:
        def _process_single_message(self, **kwargs):
            ids = torch.tensor([1, 2, 3])
            return ids, torch.ones_like(ids), torch.ones_like(ids), {}

        def sanity_check(self, *args):
            raise AssertionError("inconsistent training serialization")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(multiturn_sft_dataset, "MultiTurnSFTDataset", FakeDataset)
    monkeypatch.setattr(chat_template, "extract_system_prompt_and_generation", lambda *a, **kw: ([], []))
    with pytest.raises(AssertionError, match="inconsistent training serialization"):
        mixture.make_token_counter(tmp_path, True)(row("one"))
