import json
from argparse import Namespace

import pytest

from scripts import mix_swe_sft_data
from scripts import swe_teacher_sft as pipeline


def test_pipeline_commands_match_current_module_interfaces(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda command, env: calls.append((list(map(str, command)), env)))
    config = pipeline.configuration()
    env = pipeline.environment(config)
    args = Namespace(labels=None, allow_unaudited_pilot=True)
    pipeline.stage("collect", tmp_path, config, env, args)
    command, _ = calls[-1]
    assert command[2] == "recipe.swe_trajectory_synthesis"
    assert "--keep-going-after-success" not in command and "--enable-thinking" in command
    assert command[command.index("--input-parquet") + 1].endswith("teacher_tasks.parquet")
    pipeline.stage("mix", tmp_path, config, env, args)
    command, _ = calls[-1]
    parsed = mix_swe_sft_data.build_parser().parse_args(command[3:])
    assert parsed.teacher_fraction == 0.3 and parsed.max_length == 106496
    assert len(parsed.exclude_task_parquet) == 2
    pipeline.stage("train", tmp_path, config, env, args)
    command, train_env = calls[-1]
    assert train_env["SFT_TRAIN_FILE"].endswith("mixed/train.parquet")
    assert train_env["SFT_DATA_MANIFEST"].endswith("mixed/mixture_report.json")
    assert train_env["SFT_VAL_BEFORE_TRAIN"] == "True"
    assert train_env["SFT_TEACHER_VAL_BEFORE_TRAIN"] == "True"
    assert train_env["SWE_AGENT_MAX_TURNS"] == "150"
    assert "+data.apply_chat_template_kwargs.enable_thinking=True" in command
    assert all(not arg.startswith("+sft.teacher_forced_val_before_train") for arg in command)


def test_teacher_environment_uses_hidden_outcomes_without_protocol_shaping(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_TRAINING_THINKING_REPEAT_AGGREGATE_ENABLED", "1")
    monkeypatch.setenv("SWE_AGENT_TRAINING_THINKING_REPEAT_CROSS_TURN_ENABLED", "1")
    env = pipeline.environment(pipeline.configuration())
    assert env["LLM_API_URL"] == "http://10.248.102.141:18090/v1"
    assert env["LLM_MODEL"] == "swe-repeat-judge"
    assert env["SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING"] == "0"
    assert env["SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD"] == "0"
    assert env["SWE_AGENT_TOOL_COMMAND_FEEDBACK_ENABLED"] == "1"
    assert env["SWE_AGENT_TOOL_COMMAND_SNAPSHOT_TIMEOUT_SECONDS"] == "5"
    assert env["SWE_AGENT_TRAINING_THINKING_REPEAT_AGGREGATE_ENABLED"] == "0"
    assert env["SWE_AGENT_TRAINING_THINKING_REPEAT_CROSS_TURN_ENABLED"] == "0"


def test_filter_keeps_explicit_unaudited_override_and_unanimous_judging(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda command, env: calls.append((command, env)))
    config = pipeline.configuration()
    pipeline.stage(
        "filter", tmp_path, config, pipeline.environment(config), Namespace(labels=None, allow_unaudited_pilot=True)
    )
    command, env = calls[-1]
    assert "--allow-unaudited-pilot" in command
    assert "--judge-vote-count" not in command  # Reuse the launcher's eight-vote default.
    assert env["SWE_SFT_SOURCE_MANIFEST"] == str(tmp_path / "quality_source/sources.json")
    assert command[command.index("--max-response-tokens") + 1] == config["response_length"]
    assert command[command.index("--max-tool-dispatches") + 1] == config["max_turns"]


def test_explicit_input_skips_mining_and_controls_attempts_and_limits(tmp_path, monkeypatch):
    selected = tmp_path / "previous-selection.parquet"
    selected.touch()
    overrides = {
        "TEACHER_INPUT_PARQUET": str(selected),
        "TEACHER_ATTEMPTS": "5",
        "TEACHER_CONCURRENCY": "8",
        "TEACHER_MAX_TURNS": "300",
        "TEACHER_RESPONSE_LENGTH": "253952",
        "TEACHER_PER_TURN_TOKENS": "16384",
        "TEACHER_INFRASTRUCTURE_RETRIES": "4",
        "TEACHER_OPERATION_TIMEOUT": "1800",
        "TEACHER_TRAJECTORY_TIMEOUT": "14400",
        "TEACHER_PROGRESS_INTERVAL": "15",
    }
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    config = pipeline.configuration()
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda command, env: calls.append(list(map(str, command))))
    pipeline.stage("mine", tmp_path, config, pipeline.environment(config), Namespace())
    assert not calls
    pipeline.stage("collect", tmp_path, config, pipeline.environment(config), Namespace())
    command = calls[-1]
    expected = {
        "--input-parquet": str(selected), "--max-attempts": "5", "--concurrency": "8",
        "--max-turns": "300", "--response-length": "253952", "--per-turn-max-tokens": "16384",
        "--max-infrastructure-retries": "4", "--operation-timeout": "1800",
        "--trajectory-timeout": "14400", "--progress-interval": "15.0",
    }
    for key, value in expected.items():
        assert command[command.index(key) + 1] == value
    assert "--keep-going-after-success" not in command
    config["keep_going_after_success"] = True
    pipeline.stage("collect", tmp_path, config, pipeline.environment(config), Namespace())
    assert "--keep-going-after-success" in calls[-1]


def test_missing_explicit_input_does_not_silently_remine(tmp_path):
    config = pipeline.configuration()
    config["input_parquet"] = str(tmp_path / "missing.parquet")
    with pytest.raises(ValueError, match="Teacher input does not exist"):
        pipeline.stage("mine", tmp_path, config, pipeline.environment(config), Namespace())


def test_preflight_rejects_output_reservation_beyond_server_context(tmp_path, monkeypatch):
    config = pipeline.configuration()
    config.update(
        source_parquet=str(tmp_path), tokenizer=str(tmp_path), tool_config=str(tmp_path),
        held_out=[], input_parquet="", max_input_tokens=245760, per_turn_tokens=32768,
    )
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        pipeline, "request_json",
        lambda *args, **kwargs: {"data": [{"id": config["model"], "max_model_len": 262144}]},
    )
    with pytest.raises(ValueError, match="TEACHER_MAX_INPUT_TOKENS.*TEACHER_PER_TURN_TOKENS"):
        pipeline.preflight(config)


def test_export_cannot_freeze_a_gracefully_interrupted_collection(tmp_path, monkeypatch):
    synthesis = tmp_path / "synthesis"
    synthesis.mkdir()
    summary = synthesis / "summary.json"
    summary.write_text(json.dumps({"status": "interrupted", "tasks": 100, "terminal": 7}))
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *args: calls.append(args))
    config = pipeline.configuration()
    with pytest.raises(ValueError, match="Collection is incomplete"):
        pipeline.stage("export", tmp_path, config, pipeline.environment(config), Namespace())
    assert not calls
    summary.write_text(json.dumps({"status": "complete", "tasks": 100, "terminal": 100}))
    pipeline.stage("export", tmp_path, config, pipeline.environment(config), Namespace())
    assert len(calls) == 1


def test_resuming_data_pipeline_reuses_a_completed_quality_run(tmp_path, monkeypatch):
    full = tmp_path / "quality/full"
    full.mkdir(parents=True)
    (full / "statistics.json").write_text("{}")
    (full / "core_train.parquet").touch()
    (full / "run_status.json").write_text(json.dumps({"status": "filter_complete"}))
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda *args: calls.append(args))
    config = pipeline.configuration()
    pipeline.stage("filter", tmp_path, config, pipeline.environment(config), Namespace())
    assert not calls


def test_main_resumes_existing_pipeline_at_new_teacher_and_judge_address(tmp_path, monkeypatch):
    original = pipeline.configuration()
    contract = tmp_path / "pipeline_config.json"
    contract.write_text(json.dumps(original))
    saved = contract.read_bytes()
    current = {**original, "base_url": "http://teacher-moved:18090"}
    monkeypatch.setattr(pipeline, "configuration", lambda: current)
    monkeypatch.setattr(pipeline.sys, "argv", ["teacher-sft", "all", "--output-root", str(tmp_path)])
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda command, env: calls.append((list(map(str, command)), env)))
    # Exercise collect and filter dispatch through the actual main guard without
    # requiring source files, collection, or a judge server.
    stage = pipeline.stage
    monkeypatch.setattr(
        pipeline, "stage",
        lambda action, *args: stage(action, *args) if action in {"collect", "filter"} else None,
    )
    pipeline.main()
    assert len(calls) == 2
    assert calls[0][1]["LLM_API_URL"] == "http://teacher-moved:18090/v1"
    assert calls[1][1]["SWE_SFT_JUDGE_URL"] == "http://teacher-moved:18090"
    assert contract.read_bytes() == saved


@pytest.mark.parametrize(
    "key,new_value",
    [("model", "different-weights"), ("response_length", 100), ("source_parquet", "/different/input.parquet")],
)
def test_endpoint_migration_keeps_other_pipeline_settings_frozen(tmp_path, monkeypatch, key, new_value):
    original = pipeline.configuration()
    contract = tmp_path / "pipeline_config.json"
    contract.write_text(json.dumps(original))
    saved = contract.read_bytes()
    current = {**original, "base_url": "http://teacher-moved:18090", key: new_value}
    monkeypatch.setattr(pipeline, "configuration", lambda: current)
    monkeypatch.setattr(pipeline.sys, "argv", ["teacher-sft", "collect", "--output-root", str(tmp_path)])
    calls = []
    monkeypatch.setattr(pipeline, "stage", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Pipeline settings changed"):
        pipeline.main()
    assert not calls
    assert contract.read_bytes() == saved
