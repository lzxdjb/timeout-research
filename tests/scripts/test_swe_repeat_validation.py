"""Experimental validation must reject mismatched executor pools before inference."""

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_swe_repeat_validation_services.py"
spec = importlib.util.spec_from_file_location("repeat_validation_services", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.mark.parametrize(
    "expected_shell,reported_shell,read_enabled,source_ok,passed",
    [
        (0, False, True, True, True),
        (1, True, True, True, True),
        (0, True, True, True, False),
        (1, False, True, True, False),
        (1, True, False, True, False),
        (1, True, True, False, False),
    ],
)
def test_preflight_checks_service_flags_source_and_credential_redaction(
    tmp_path, monkeypatch, expected_shell, reported_shell, read_enabled, source_ok, passed
):
    names = [
        "remote_execution_service.py",
        "tools.py",
        "tool_repeat_evidence.py",
        "tool_contract_info.py",
        "tool_arguments.py",
        "test_results.py",
        "python_test_repeat_evidence.py",
    ]
    directory = tmp_path / "recipe/swe_agent"
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_text(name)
    hashes = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
    if not source_ok:
        hashes["tools.py"] = "old source"
    info = {
        "ok": True,
        "source_fingerprint": {"files": hashes},
        "private_diagnostics": {
            "repeat_read_evidence_enabled": read_enabled,
            "repeat_shell_evidence_enabled": reported_shell,
        },
    }

    class Opener:
        def open(self, request, timeout):
            assert request.headers["Authorization"] == "Bearer test-secret"
            return io.BytesIO(json.dumps(info).encode())

    monkeypatch.setattr(preflight, "build_opener", lambda *_: Opener())
    monkeypatch.setenv("SWE_SOURCE", str(tmp_path))
    monkeypatch.setenv("SWE_AGENT_EXECUTION_URLS", "http://offline-pool")
    monkeypatch.setenv("SWE_AGENT_EXECUTION_TOKEN", "test-secret")
    output = tmp_path / "services.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--expect-shell", str(expected_shell), "--output", str(output)])
    if passed:
        preflight.main()
    else:
        with pytest.raises(SystemExit) as exc:
            preflight.main()
        assert exc.value.code == 2
    text = output.read_text()
    assert json.loads(text)["passed"] is passed
    assert "test-secret" not in text and "Authorization" not in text


@pytest.mark.parametrize(
    "arm,python_shadow,shell,penalty",
    [
        ("baseline", "0", "0", "0"),
        ("python", "1", "0", "0"),
        ("shell", "0", "1", "0"),
        ("both", "1", "1", "0"),
        ("python-penalty", "1", "0", "1"),
        ("both-penalties", "1", "1", "1"),
    ],
)
def test_validation_arms_explicitly_select_penalty_without_launching(
    tmp_path,
    arm,
    python_shadow,
    shell,
    penalty,
):
    # Intercept the final exec, never source launchers or contact services.
    stub = tmp_path / "bash"
    stub.write_text(
        "#!" + sys.executable + "\nimport json, os\n"
        "print(json.dumps({k: os.environ[k] for k in ("
        "'SWE_AGENT_TOOL_REPEAT_PYTHON_TEST_SHADOW_ENABLED',"
        "'SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_SHELL_READS',"
        "'SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TESTS')}))\n"
    )
    stub.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        CONFIG_ONLY="1",
        SWE_AGENT_EXECUTION_URLS="http://unused-offline",
        EXPERIMENT_NAME="offline_test_" + tmp_path.name,
        SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TESTS="1",
    )
    runner = SCRIPT.parent.parent / "running_script/run_repeat_validation_ablation.sh"
    completed = subprocess.run(
        ["/bin/bash", str(runner), arm], env=environment, text=True, capture_output=True, check=True
    )
    values = json.loads(completed.stdout)
    assert list(values.values()) == [python_shadow, shell, penalty]


@pytest.mark.parametrize("penalty", [None, "1"])
def test_shared_launcher_forwards_default_off_and_explicit_actor_overrides(tmp_path, penalty):
    root = SCRIPT.parent.parent
    stub = tmp_path / "bash"
    stub.write_text("#!" + sys.executable + "\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        TRAIN_FILES='["offline.parquet"]',
        VAL_FILES='["offline.parquet"]',
        SWE_AGENT_EXECUTION_URLS="http://unused-offline",
        TARGET_VERL=str(root),
        SWE_SOURCE=str(root.parent / "stock-rl-reflect"),
        MODEL_PATH=str(tmp_path),
        CONFIG_ONLY="1",
    )
    for name in (
        "SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TESTS",
        "SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TEST_PROTECTED_CALLS",
        "SWE_SOURCE_EXPECTED_COMMIT",
    ):
        environment.pop(name, None)
    if penalty is not None:
        environment["SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TESTS"] = penalty
    completed = subprocess.run(
        ["/bin/bash", str(root / "scripts/run_swe_agent_qwen3_5_35b_a3b.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    overrides = json.loads(completed.stdout.strip().splitlines()[-1])
    assert "actor_rollout_ref.actor.tool_repeat_penalty_python_tests=" + (penalty or "0") in overrides
    assert "actor_rollout_ref.actor.tool_repeat_penalty_python_test_protected_calls=20" in overrides
