import json

import pytest

from scripts import diagnose_trajectories as diagnosis


def call(name="Read", **parameters):
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>" for key, value in parameters.items()
    )
    return f"<function={name}>\n{body}</function>"


def output_with_calls(*calls):
    parts = []
    for index, value in enumerate(calls):
        if index:
            parts.append("assistant\n")
        parts.append(value)
        parts.append("\nuser\n<tool_response>ok</tool_response>\n")
    parts.append("assistant\nDone.")
    return "".join(parts)


def row(*, step=1, line_id="task", raw=1.0, shaped=1.0, output=None, **extra):
    value = {
        "step": step,
        "uid": line_id,
        "gts": {"benchmark": "swe_rebench_v2", "instance_id": line_id, "repo": "owner/repo"},
        "raw_score": raw,
        "shaped_score": shaped,
        "output": output if output is not None else output_with_calls(call(file_path="src/a.py")),
        "trajectory_tool_dispatches": 1,
        "trajectory_response_tokens": 12,
        "trajectory_assistant_turns": 2,
        "trajectory_penalized_repeated_tool_calls": 0,
        "trajectory_repeated_tool_calls": 0,
        "trajectory_tool_error_returns": 0,
    }
    value.update(extra)
    return value


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


def test_explicit_file_excludes_padding_and_reports_step_manifest(tmp_path):
    path = tmp_path / "1.jsonl"
    write_jsonl(path, [row(line_id="a"), {"is_padding": True}, row(line_id="b")])

    records, manifest = diagnosis.read_input(
        diagnosis.InputSpec(path, "train-step-1"),
        read_window=8,
        exec_window=16,
        transcript_chars=1000,
    )

    assert len(records) == 2
    assert manifest["padding_rows_excluded"] == 1
    assert manifest["step_from_filename"] == 1
    assert manifest["row_step_mismatches"] == 0


def test_expected_cohort_and_reasoning_loops_are_independent_of_tool_counts(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    dataset = tmp_path / "expected.parquet"
    pq.write_table(pa.Table.from_pylist([{"extra_info": {"task_id": "loop"}}, {"extra_info": {"task_id": "missing"}}]), dataset)
    source = tmp_path / "0.jsonl"
    repeated = "This is a long repeated reasoning paragraph about the same problem and the same unchanged hypothesis without progress.\n\n" * 200
    write_jsonl(source, [row(step=0, line_id="loop", output="assistant\n<think>\n" + repeated)])
    result = diagnosis.diagnose([diagnosis.InputSpec(source, "run")], tmp_path / "out", expected_dataset=dataset)
    assert result["expected_cohort_coverage"]["missing"] == {"missing": 1}
    assert result["expected_cohort_coverage"]["status"] == "incomplete"
    metrics = result["groups"][0]["metrics"]
    assert metrics["repeated_assistant_text_count"] == 1 and metrics["large_assistant_turn_count"] == 1
    write_jsonl(source, [row(step=0, line_id="loop"), row(step=0, line_id="missing")])
    result = diagnosis.diagnose([diagnosis.InputSpec(source, "run")], tmp_path / "complete", expected_dataset=dataset)
    assert result["expected_cohort_coverage"]["status"] == "complete"


def test_failure_history_does_not_let_unrelated_pass_erase_failure(tmp_path):
    text = (call("Bash", command="pytest a.py") + '\nuser\n<tool_response>exit_code: 1\n1 failed, 2 warnings in 1s</tool_response>\nassistant\n'
            + call("Bash", command="pytest b.py") + '\nuser\n<tool_response>exit_code: 0\n1 passed in 1s</tool_response>\nassistant\nDone.')
    source = tmp_path / "0.jsonl"
    write_jsonl(source, [row(step=0, output=text, trajectory_tool_dispatches=3, submission_signal_seen=True)])
    diagnosis.diagnose([diagnosis.InputSpec(source, "run")], tmp_path / "out")
    record = json.loads((tmp_path / "out/all_trajectories.jsonl").read_text())
    assert record["behavior"]["unresolved_public_failure_events"] == [0]
    assert record["behavior"]["unresolved_failure_at_submission"]
    assert record["behavior"]["budget_boundary_discrepancy"]
    assert [v["outcome"] for v in record["behavior"]["public_runner_history"]] == ["failed", "passed"]


def test_replayed_runner_signals_include_deselection_and_native_sympy():
    deselected = diagnosis._response_signals("Bash", {"command": "pytest -k absent"},
        "exit_code: 5\n199 deselected in 0.27s")
    assert deselected["zero_test_execution"]
    native = diagnosis._response_signals("Bash", {"command": "python bin/test sympy/polys/tests/test_monomials.py"},
        "exit_code: 0\n=== tests finished: 11 passed, in 0.07 seconds ===")
    assert native["public_test_command_candidate"]
    assert native["test_summary_replay"]["tests_executed"] == 11


@pytest.mark.parametrize("fault", [None, "corrupt", "invalid_deflate", "overflow", "escape"])
def test_private_full_log_collection_is_bounded_verified_and_private(tmp_path, fault):
    import gzip
    import hashlib
    private = tmp_path / "private"
    private.mkdir()
    raw = b"BEGIN\n" + b"secret hidden details\n" * 1000 + b"END"
    compressed = gzip.compress(raw, mtime=0)
    if fault == "invalid_deflate":
        # Valid gzip header with a forbidden DEFLATE block type, correctly hashed.
        compressed = compressed[:10] + b"\x07" + compressed[-8:]
    blob_digest = hashlib.sha256(compressed).hexdigest()
    name = blob_digest + ".log.gz"
    blob = private / name
    blob.write_bytes(compressed if fault != "corrupt" else b"wrong")
    if fault == "escape":
        blob.unlink()
        outside = tmp_path / "outside"
        outside.write_bytes(compressed)
        blob.symlink_to(outside)
    ref = {"file": name, "sha256": blob_digest, "compression": "gzip", "bytes": len(compressed),
           "uncompressed_bytes": len(raw), "original_bytes": len(raw) + (100 if fault == "overflow" else 0),
           "uncompressed_sha256": hashlib.sha256(raw).hexdigest(), "complete": fault != "overflow"}
    artifact = {"task_id": "task", "phases": {"test_execution": {"sha256": hashlib.sha256(raw).hexdigest(), "truncated": True, "full_log": ref}}}
    data = json.dumps(artifact).encode()
    digest = hashlib.sha256(data).hexdigest()
    (private / (digest + ".json")).write_bytes(data)
    record = {"task_id": "task", "reward_evidence": {"private_evaluator_artifact_sha256": digest,
        "private_evaluator_artifact_path": "/service/" + digest + ".json"}}
    result = diagnosis.collect_private_evaluator(record, ["/service=" + str(private)], tmp_path / "out")
    if fault in {"corrupt", "escape", "invalid_deflate"}:
        assert result["status"] in {"log_hash_mismatch", "rejected_log_path", "invalid_or_missing_log"}
        assert not (tmp_path / "out/private_evaluator").exists()
    else:
        assert result["status"] == "verified" and result["logs_complete"] == (fault is None)
        copied = tmp_path / "out/private_evaluator" / name
        assert gzip.decompress(copied.read_bytes()) == raw
        assert copied.stat().st_mode & 0o077 == 0
        assert "secret hidden details" not in json.dumps(result)


def test_completion_cutoff_log_is_distinct_from_saved_and_eligible_denominators(tmp_path):
    path = tmp_path / '0.jsonl'
    write_jsonl(path, [row(step=0, reward_eligible=1), row(step=0, line_id='ineligible', raw=0, reward_eligible=0)])
    log = tmp_path / 'launcher.log'
    log.write_text('private launcher content\nApplied val completion-ratio cutoff: terminal=2 total=3 threshold=0.9 requested=1 cancelled=1\n')
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'cohort')], tmp_path / 'out', run_log=log)
    assert report['completion_coverage']['cutoffs'][0]['total'] == 3
    assert report['completion_coverage']['saved_rows'] == 2
    assert report['groups'][0]['metrics']['cohorts']['reward_eligible']['rows'] == 1
    assert 'private launcher content' not in json.dumps(report)
    assert 'private launcher content' not in (tmp_path / 'out' / 'report.md').read_text()


def test_private_artifact_collection_is_mapped_verified_and_not_in_public_report(tmp_path):
    import hashlib
    private = tmp_path / "private"
    private.mkdir()
    data = json.dumps({"task_id": "task", "service_id": "service-1", "failure_phase": "verifier_setup", "phases": {"setup": "hidden secret"}}).encode()
    digest = hashlib.sha256(data).hexdigest()
    artifact = private / (digest + ".json")
    artifact.write_bytes(data)
    source = tmp_path / "0.jsonl"
    write_jsonl(source, [row(step=0, private_evaluator_artifact_path="/service-private/" + artifact.name,
        private_evaluator_artifact_sha256=digest, private_evaluator_service_id="service-1",
        infrastructure_failure_reason="silent_command_timeout", observed_infrastructure_failure_reason="swe_terminal_tool_failure")])
    report = diagnosis.diagnose([diagnosis.InputSpec(source, "case")], tmp_path / "out",
                               evaluator_roots=["/service-private=" + str(private)])
    metrics = report["groups"][0]["metrics"]
    assert metrics["private_evaluator_collection"] == {"verified": 1}
    assert metrics["causal_reason_policy_conflict_count"] == 1
    assert "hidden secret" not in json.dumps(report)
    copied = tmp_path / "out/private_evaluator" / artifact.name
    assert copied.read_bytes() == data and copied.stat().st_mode & 0o077 == 0
    artifact.write_text("corrupt")
    report = diagnosis.diagnose([diagnosis.InputSpec(source, "case")], tmp_path / "bad",
                               evaluator_roots=["/service-private=" + str(private)])
    assert report["groups"][0]["metrics"]["private_evaluator_collection"] == {"hash_mismatch": 1}


@pytest.mark.parametrize("fault,expected", [("symlink", "rejected_path"), ("task", "task_mismatch"),
    ("service", "service_mismatch"), ("type", "invalid_json"), ("missing", "not_accessible")])
def test_private_evidence_collection_rejects_wrong_identity_and_escaped_paths(tmp_path, fault, expected):
    import hashlib
    private = tmp_path / "private"
    private.mkdir()
    body = {"task_id": "wrong" if fault == "task" else "task", "service_id": "wrong" if fault == "service" else "service"}
    data = json.dumps([] if fault == "type" else body).encode()
    digest = hashlib.sha256(data).hexdigest()
    artifact = private / (digest + ".json")
    if fault == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(data)
        artifact.symlink_to(outside)
    elif fault != "missing":
        artifact.write_bytes(data)
    record = {"task_id": "task", "reward_evidence": {"private_evaluator_artifact_path": "/service/" + artifact.name,
        "private_evaluator_artifact_sha256": digest, "private_evaluator_service_id": "service"}}
    result = diagnosis.collect_private_evaluator(record, ["/service=" + str(private)], tmp_path / "out")
    assert result["status"] == expected
    assert not (tmp_path / "out/private_evaluator").exists()


def test_diagnose_rejects_mixed_steps_in_one_explicit_file(tmp_path):
    path = tmp_path / "1.jsonl"
    write_jsonl(path, [row(step=1, line_id="a"), row(step=2, line_id="b")])

    with pytest.raises(ValueError, match="mixed trajectory steps"):
        diagnosis.diagnose([diagnosis.InputSpec(path, "mixed")], tmp_path / "out")


def test_report_reconstructs_repeats_and_bounds_transcript(tmp_path):
    repeated = output_with_calls(
        call("Read", file_path="src/a.py"),
        call("Read", file_path="src/a.py"),
    ) + "x" * 1000
    path = tmp_path / "1.jsonl"
    write_jsonl(path, [row(line_id="repeat", output=repeated, trajectory_tool_dispatches=2)])

    report = diagnosis.diagnose(
        [diagnosis.InputSpec(path, "grpo")],
        tmp_path / "out",
        transcript_chars=100,
        examples_per_category=1,
    )

    group = report["groups"][0]
    assert group["metrics"]["repeat_event_rate"] == 1.0
    assert group["metrics"]["stored_replay_mismatch_count"] == 1
    selected = group["selected"]["highest_penalized_repeats"][0]
    assert selected["reconstructed"]["penalized_repeats"] == 1
    assert len(selected["transcript_excerpt"]) <= 100 + len("\n...[middle truncated]...\n")
    assert (tmp_path / "out" / "report.md").is_file()
    assert (tmp_path / "out" / "selected_trajectories.jsonl").is_file()


def test_representative_categories_are_deterministic(tmp_path):
    path = tmp_path / "1.jsonl"
    write_jsonl(
        path,
        [
            row(line_id="success-repeat", raw=1.0, output=output_with_calls(call(file_path="a"), call(file_path="a")), trajectory_tool_dispatches=2),
            row(line_id="zero-clean", raw=0.0, output=output_with_calls(call(file_path="b")), trajectory_tool_dispatches=1),
            row(line_id="error", output=output_with_calls(call(file_path="c")), trajectory_tool_dispatches=1, trajectory_tool_error_returns=1),
        ],
    )
    report = diagnosis.diagnose([diagnosis.InputSpec(path, "step-1")], tmp_path / "out", examples_per_category=1)
    selected = report["groups"][0]["selected"]
    assert selected["raw_success_with_repeats"][0]["task_id"] == "success-repeat"
    assert selected["zero_reward_without_repeats"][0]["task_id"] == "zero-clean"
    assert selected["tool_errors"][0]["task_id"] == "error"


def test_parser_failures_are_reported_without_dropping_row(tmp_path):
    path = tmp_path / "2.jsonl"
    write_jsonl(path, [row(step=2, line_id="bad", raw=0.0, output="assistant\n")])

    report = diagnosis.diagnose([diagnosis.InputSpec(path, "validation")], tmp_path / "out")

    group = report["groups"][0]
    assert group["metrics"]["parser_failure_count"] == 1
    assert group["metrics"]["reconstruction_unknown_count"] == 1
    assert group["metrics"]["zero_reward_without_repeats"] == 0
    assert not group["selected"]["zero_reward_without_repeats"]
    assert group["selected"]["parser_failures"][0]["task_id"] == "bad"


def test_unknown_tool_names_are_reported_as_interface_drift(tmp_path):
    path = tmp_path / "1.jsonl"
    output = output_with_calls(call("complete_writeValuesLogic", file_path="src/a.py"))
    write_jsonl(path, [row(line_id="unknown", raw=0.0, output=output)])

    report = diagnosis.diagnose([diagnosis.InputSpec(path, "train")], tmp_path / "out")

    group = report["groups"][0]
    assert group["metrics"]["unknown_tool_call_count"] == 1
    assert group["metrics"]["unknown_tool_trajectory_count"] == 1
    assert group["metrics"]["unknown_tool_names"] == ["complete_writeValuesLogic"]
    assert group["selected"]["unknown_tools"][0]["task_id"] == "unknown"


def test_payload_role_marker_is_analysis_unknown_not_clean_trajectory(tmp_path):
    path = tmp_path / '1.jsonl'
    write_jsonl(path, [row(raw=0, output=output_with_calls(call('search_files', query='user')))])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'collision')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['payload_role_marker_trajectory_count'] == 1
    assert metrics['reconstruction_unknown_count'] == 1
    assert metrics['repeat_rate_denominator'] == 0
    assert metrics['repeat_event_rate'] is None
    assert metrics['stored_replay_mismatch_count'] == 0


def test_changed_search_limits_still_surface_identical_observations(tmp_path):
    path = tmp_path / '1.jsonl'
    output = output_with_calls(
        call('search_files', query='set', max_results=20),
        call('search_files', query='set', max_results=100000000),
    )
    write_jsonl(path, [row(output=output, trajectory_tool_dispatches=2)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'search')], tmp_path / 'out')
    selected = report['groups'][0]['selected']['repeated_search_observations'][0]
    assert selected['reconstructed']['penalized_repeats'] == 0
    assert selected['reconstructed']['same_search_observation_count'] == 1


def test_empty_patch_success_requires_explicit_metadata(tmp_path):
    path = tmp_path / '1.jsonl'
    write_jsonl(path, [row(final_patch_empty=1), row(line_id='missing'), row(line_id='unknown', final_patch_empty=-1)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'reward')], tmp_path / 'out')
    assert report['groups'][0]['metrics']['success_with_empty_patch_count'] == 1
    assert len((tmp_path / 'out' / 'all_trajectories.jsonl').read_text().splitlines()) == 3


def test_empty_output_is_unknown_and_full_observation_hashes_differ(tmp_path):
    path = tmp_path / '1.jsonl'
    prefix = 'x' * 600
    output = output_with_calls(call('search_files', query='x'), call('search_files', query='x'))
    output = output.replace('<tool_response>ok', '<tool_response>' + prefix + 'a', 1)
    output = output.replace('<tool_response>ok', '<tool_response>' + prefix + 'b', 1)
    write_jsonl(path, [row(output=''), row(line_id='different', output=output)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'boundary')], tmp_path / 'out')
    assert report['groups'][0]['metrics']['reconstruction_unknown_count'] == 1
    assert report['groups'][0]['metrics']['same_search_observation_count'] == 0


def test_declared_schema_wins_over_global_aliases_and_manual_detail_is_lossless(tmp_path):
    path = tmp_path / '1.jsonl'
    original = row(line_id='schema', output=output_with_calls(call('Read', file_path='a')))
    original['input'] = '<tools>\n' + json.dumps({'function': {'name': 'read_file'}}) + '\n</tools>'
    write_jsonl(path, [original, row(line_id='manual')])
    report = diagnosis.diagnose(
        [diagnosis.InputSpec(path, 'train')], tmp_path / 'out', examples_per_category=1, inspect=['train:2'],
    )
    assert report['groups'][0]['metrics']['unknown_tool_names'] == ['Read']
    detail = list((tmp_path / 'out' / 'details').glob('*line2.json'))
    assert len(detail) == 1
    assert json.loads(detail[0].read_text())['uid'] == 'manual'
    assert report['diagnosis_source_sha256']


def test_zero_exit_with_compiler_failure_is_review_signal(tmp_path):
    path = tmp_path / '1.jsonl'
    output = output_with_calls(call('Bash', command='cargo build 2>&1 | tail -20'))
    output = output.replace('<tool_response>ok', '<tool_response>exit_code: 0\nerror: could not compile crate')
    write_jsonl(path, [row(output=output)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'masked')], tmp_path / 'out')
    assert report['groups'][0]['metrics']['zero_exit_failure_text_count'] == 1
    assert report['groups'][0]['metrics']['final_patch_metadata_known_count'] == 0


def test_claim_failure_is_not_model_parser_failure_and_cohorts_are_separate(tmp_path):
    path = tmp_path / '0.jsonl'
    write_jsonl(path, [
        row(step=0, line_id='admitted', reward_eligible=1),
        row(step=0, line_id='claim', raw=0, shaped=0, output='', reward_eligible=0,
            infrastructure_failure=0, infrastructure_failure_reason='claim_failure',
            observed_infrastructure_failure=1, observed_infrastructure_failure_reason='validation_binary',
            trajectory_tool_dispatches=0),
        row(step=0, line_id='malformed', raw=0, output='assistant\n', reward_eligible=1),
    ])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'claims')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['pre_generation_failure_count'] == 1
    assert metrics['parser_failure_count'] == 1
    assert metrics['observed_infrastructure_failure_count'] == 1
    assert metrics['infrastructure_failure_count'] == 0
    assert metrics['infrastructure_failure_reasons'] == {'claim_failure': 1}
    assert metrics['cohorts']['reward_eligible']['raw_success_rate'] == 0.5
    assert metrics['raw_success_rate'] == pytest.approx(1 / 3)
    assert report['groups'][0]['selected']['pre_generation_failures'][0]['parser_error'] == ''
    assert metrics['causal_reason_policy_conflict_count'] == 1


def test_content_identity_survives_new_capsules_and_changed_call_limits(tmp_path):
    artifact_hash = 'a' * 64
    output = output_with_calls(call('Bash', command='git log | head -100'), call('Bash', command='git log | head -200'))
    for index in range(2):
        response = f'exit_code: 0\n[tool output stored]\nartifact_path: .swe_agent/tool_outputs/{index}.log\nartifact_sha256: {artifact_hash}\nretrieval: Grep then Read\n'
        output = output.replace('<tool_response>ok', '<tool_response>' + response, 1)
    original = row(output=output, trajectory_tool_dispatches=3)
    original['output'] = original['output'].removesuffix('Done.') + call('Bash', command='git log | head -300')
    original['input'] = '<tools>\n' + '\n'.join(json.dumps({'function': {'name': name}}) for name in ('Bash', 'Grep', 'Read')) + '\n</tools>'
    path = tmp_path / '1.jsonl'
    write_jsonl(path, [original])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'artifact')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['repeated_artifact_content_count'] == 1
    assert metrics['artifact_schema_mismatch_count'] == 0
    assert metrics['unpaired_tool_call_count'] == 1
    selected = report['groups'][0]['selected']['repeated_artifact_content'][0]
    assert selected['artifact_content_groups'][0]['events'] == [0, 1]
    assert selected['events'][0]['shell_exit_code_in_response'] == 0


def test_advisory_warning_continuation_and_schema_errors_are_separate(tmp_path):
    output = output_with_calls(call('Bash', command='false'), call('Bash', command='false'), call('Bash', command='false'), call('Grep', pattern='x', limit=2))
    for response in ('exit_code: 1\n', 'exit_code: 1\nThis command failed again with the same output.',
                     'exit_code: 1\n[tool feedback] This exact call returned the same observation 3 times.',
                     'Search error: unsupported parameters: limit.'):
        output = output.replace('<tool_response>ok', '<tool_response>' + response, 1)
    path = tmp_path / '1.jsonl'
    write_jsonl(path, [row(output=output)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'feedback')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['advisory_warning_count'] == 2
    assert metrics['continued_after_warning_count'] == 1
    assert metrics['search_schema_error_count'] == 1
    assert metrics['search_backend_error_count'] == 0
    assert metrics['search_error_reasons'] == {'search_schema_error': 1}
    assert report['groups'][0]['selected']['continued_after_warning'][0]['events'][2]['continued_after_warning']


def test_recorded_cutoffs_are_not_inferred_from_historical_dispatch_limit(tmp_path):
    path = tmp_path / '1.jsonl'
    write_jsonl(path, [row(trajectory_tool_dispatches=149), row(line_id='flag', trajectory_budget_reached=True, trajectory_max_turn=1)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'cutoff')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['budget_or_149_call_cap_count'] == 2
    assert metrics['recorded_budget_reached_count'] == 1
    assert metrics['recorded_max_turn_cutoff_count'] == 1


def test_artifact_failure_summary_and_schema_mismatch_use_complete_response(tmp_path):
    path = tmp_path / '1.jsonl'
    response = '\n[tool output stored]\nretrieval: Use search_text then read_file\n' + 'x' * 700
    response += '\nimportant_lines:\nexit_code: 0\n1 failed, 64 passed\n'
    output = output_with_calls(call('Bash', command='python -m pytest tests | tail -20'))
    output = output.replace('<tool_response>ok', '<tool_response>' + response)
    original = row(output=output)
    original['input'] = '<tools>\n' + json.dumps({'function': {'name': 'Bash'}}) + '\n</tools>'
    write_jsonl(path, [original])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'capsule')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['zero_exit_failure_text_count'] == 1
    assert metrics['artifact_schema_mismatch_count'] == 1
    assert metrics['public_test_command_candidate_count'] == 1
    assert metrics['explicit_verification_call_count'] == 0
    event = report['groups'][0]['selected']['zero_exit_failure_text'][0]['events'][0]
    assert event['shell_pipeline'] is True
    assert event['shell_exit_code_in_response'] == 0
    assert '1 failed' not in event['response']


def test_no_op_edit_and_unicode_backend_errors_are_distinct_signals(tmp_path):
    path = tmp_path / '1.jsonl'
    output = output_with_calls(
        call('Grep', pattern='class TemplateView'),
        call('Edit', file_path='a.py', old_string='x', new_string='x'),
    )
    output = output.replace('<tool_response>ok', '<tool_response>Search error: non-ASCII case-insensitive search requires rg', 1)
    output = output.replace('<tool_response>ok', '<tool_response>Edit made no changes: old_string and new_string are identical.', 1)
    write_jsonl(path, [row(output=output)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'signals')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['search_error_reasons'] == {'unicode_backend_limitation': 1}
    assert metrics['no_op_edit_count'] == 1
    assert metrics['zero_exit_failure_text_count'] == 0


def test_identical_observation_groups_use_full_hash_and_do_not_claim_no_mutation(tmp_path):
    path = tmp_path / '1.jsonl'
    output = output_with_calls(call('Read', file_path='a'), call('Read', file_path='a'), call('Read', file_path='a'))
    output = output.replace('<tool_response>ok', '<tool_response>' + 'x' * 600 + 'a', 2)
    output = output.replace('<tool_response>ok', '<tool_response>' + 'x' * 600 + 'b', 1)
    write_jsonl(path, [row(output=output)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'hash')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['identical_call_observation_count'] == 1
    selected = report['groups'][0]['selected']['identical_call_observations'][0]
    assert selected['identical_call_observation_groups'][0]['events'] == [0, 1]
    assert 'workspace_unchanged' not in selected['identical_call_observation_groups'][0]


def test_failure_text_does_not_override_nonzero_or_clean_exit(tmp_path):
    path = tmp_path / '1.jsonl'
    output = output_with_calls(call('Bash', command='pytest'), call('Bash', command='pytest'))
    output = output.replace('<tool_response>ok', '<tool_response>exit_code: 1\n1 failed', 1)
    output = output.replace('<tool_response>ok', '<tool_response>exit_code: 0\n64 passed', 1)
    write_jsonl(path, [row(output=output)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'statuses')], tmp_path / 'out')
    assert report['groups'][0]['metrics']['zero_exit_failure_text_count'] == 0


def test_explicit_verification_is_separate_from_detector_inference(tmp_path):
    path = tmp_path / '1.jsonl'
    output = output_with_calls(call('Bash', command='pytest'), call('Bash', command='pytest', verification='true'))
    write_jsonl(path, [row(output=output)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'verification')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['detector_verification_call_count'] == 2
    assert metrics['explicit_verification_call_count'] == 1


def test_runner_replay_categories_keep_historical_outcomes_and_private_provenance(tmp_path):
    path = tmp_path / '0.jsonl'
    commands = ['pytest | head -100', 'python tests/runtests.py', 'pytest -xvs', 'git log -- src/_pytest/python.py']
    observations = [
        'exit_code: 0\n================ FAILURES ================\nE   AssertionError: bad\n[test outcome] unknown',
        'exit_code: 0\nRan 8 tests in 1s\nFAILED (errors=138)\n[test outcome] failed',
        'exit_code: 0\n================ FAILURES ================\n2 failed in 0.1s\n================ 1 passed in 0.2s ================\n[test outcome] passed',
        'exit_code: 0\ncommit pytest fixture',
    ]
    output = output_with_calls(*(call('Bash', command=command) for command in commands))
    for observation in observations:
        output = output.replace('<tool_response>ok', '<tool_response>' + observation, 1)
    write_jsonl(path, [row(step=0, output=output, hidden_failure_phase='verifier_setup',
        private_evaluator_artifact_path='/private/evidence.json', legacy_patch_metrics_source='worktree_fingerprint',
        final_worktree_hashed_bytes=1000000, final_tracked_diff_bytes=200, final_patch_changed_bytes=1000000)])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'runner')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['incomplete_test_evidence_count'] == 1
    assert metrics['conflicting_test_counts_count'] == 1
    assert metrics['nested_test_summaries_count'] == 1
    assert metrics['implicit_test_attempt_count'] == 3
    selected = report['groups'][0]['selected']
    assert selected['evaluator_phase_failures'] and selected['worktree_patch_provenance']
    events = selected['nested_test_summaries'][0]['events']
    assert events[2]['reported_test_outcome'] == 'passed'
    assert events[2]['test_summary_replay']['test_outcome'] == 'passed'
    assert events[2]['test_summary_replay_provenance'] == 'current_parser_on_saved_response'
    assert events[3]['test_summary_replay']['test_framework'] == 'unknown'
    assert selected['evaluator_phase_failures'][0]['reward_evidence']['private_evaluator_artifact_path'] == '/private/evidence.json'


def test_environment_zero_tests_and_literal_escape_candidates(tmp_path):
    path = tmp_path / '1.jsonl'
    output = output_with_calls(
        call('Bash', command='python -m pytest'),
        call('Bash', command='python tests/runtests.py bad'),
        call('Grep', pattern=r'Item\(', literal='true'),
    )
    output = output.replace('<tool_response>ok', '<tool_response>exit_code: 1\nNo module named pytest', 1)
    output = output.replace('<tool_response>ok', '<tool_response>exit_code: 0\nRan 0 tests in 0.000s\nOK', 1)
    output = output.replace('<tool_response>ok', '<tool_response>No matches.', 1)
    write_jsonl(path, [row(output=output), row(line_id='discussion', output='Explaining the issue without using tools.')])
    report = diagnosis.diagnose([diagnosis.InputSpec(path, 'environment')], tmp_path / 'out')
    metrics = report['groups'][0]['metrics']
    assert metrics['environment_error_counts'] == {'pytest_unavailable': 1}
    assert metrics['zero_test_execution_count'] == 1
    assert metrics['literal_regex_escape_candidate_count'] == 1
    assert metrics['generated_without_dispatch_count'] == 1
