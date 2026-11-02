"""Run the existing SWE mining, synthesis, quality-filter and SFT modules.

Run ``python -m scripts.swe_teacher_sft --help`` from the verl checkout.
The ``all`` action builds data only; ``train`` explicitly starts SFT.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERL = Path(__file__).resolve().parents[1]
SOURCE = VERL.parent / "stock-rl-reflect"
REFERENCE = SOURCE / "data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet"
VALIDATION = SOURCE / "data/swe_benchmarks/verl/swe_bench_verified_reference_verified_repeat2_500_images_v1.parquet"
VALIDATION_94 = (
    SOURCE
    / "data/swe_benchmarks/verl-100-only-in-yatelanda-dsw"
    / "swe_bench_verified_reset_v9_identity_v1_reference_verified_repeat2.parquet"
)
STUDENT = SOURCE / "data/3552-checkpoint"
EXISTING = VERL / "analysis/swe_sft_quality/v5_unaudited/snapshots/checkpoint_001/core_train.parquet"


def setting(name: str, default: object) -> str:
    return os.environ.get(name) or str(default)


def resume_configuration(config: dict) -> dict:
    """Freeze experiment settings while allowing the same server to relocate.

    Keep the original pipeline_config.json for provenance. The current address
    is passed to both generation and quality judging on each invocation.
    """
    return {key: value for key, value in config.items() if key != "base_url"}


def configuration() -> dict:
    return {
        "version": 2,
        "input_parquet": setting("TEACHER_INPUT_PARQUET", ""),
        "source_parquet": setting("TEACHER_RL_SOURCE", REFERENCE),
        "rollout_dir": setting("TEACHER_RL_ROLLOUT_DIR", VERL / "rollout_data/three_penalty_train_0_96"),
        "min_step": int(setting("TEACHER_MIN_STEP", 20)),
        "max_step": int(setting("TEACHER_MAX_STEP", 29)),
        "min_attempts": int(setting("TEACHER_MIN_VALID_ATTEMPTS", 8)),
        "min_distinct_steps": int(setting("TEACHER_MIN_DISTINCT_STEPS", 1)),
        "max_tasks": int(setting("TEACHER_TASK_LIMIT", 100)),
        "model": setting("TEACHER_MODEL", "swe-repeat-judge"),
        "base_url": setting("TEACHER_BASE_URL", "http://10.248.102.141:18090").rstrip("/"),
        "tokenizer": setting("TEACHER_STUDENT_TOKENIZER", STUDENT),
        "tool_config": setting("TEACHER_TOOL_CONFIG", SOURCE / "recipe/swe_agent/config/tool_config_efficient_v1.yaml"),
        "max_attempts": int(setting("TEACHER_ATTEMPTS", 3)),
        "keep_going_after_success": setting("TEACHER_KEEP_GOING_AFTER_SUCCESS", "0") == "1",
        "max_infrastructure_retries": int(setting("TEACHER_INFRASTRUCTURE_RETRIES", 3)),
        "concurrency": int(setting("TEACHER_CONCURRENCY", 4)),
        "temperature": float(setting("TEACHER_TEMPERATURE", 0.7)),
        "seed": int(setting("TEACHER_SEED", 42)),
        "max_turns": int(setting("TEACHER_MAX_TURNS", 120)),
        "response_length": int(setting("TEACHER_RESPONSE_LENGTH", 95536)),
        "per_turn_tokens": int(setting("TEACHER_PER_TURN_TOKENS", 8192)),
        "max_input_tokens": int(setting("TEACHER_MAX_INPUT_TOKENS", 245760)),
        "operation_timeout": int(setting("TEACHER_OPERATION_TIMEOUT", 1200)),
        "trajectory_timeout": int(setting("TEACHER_TRAJECTORY_TIMEOUT", 7200)),
        "progress_interval": float(setting("TEACHER_PROGRESS_INTERVAL", 30)),
        "enable_thinking": setting("TEACHER_ENABLE_THINKING", "1") == "1",
        "existing_train": setting("TEACHER_EXISTING_SFT", EXISTING),
        "teacher_fraction": float(setting("TEACHER_MIX_FRACTION", 0.3)),
        "val_fraction": float(setting("TEACHER_SFT_VAL_FRACTION", 0.05)),
        "max_length": int(setting("SFT_MAX_LENGTH", 106496)),
        "held_out": json.loads(setting("TEACHER_HELD_OUT_FILES", json.dumps([str(VALIDATION), str(VALIDATION_94)]))),
    }


def environment(config: dict) -> dict:
    result = os.environ.copy()
    result.update(
        PYTHONPATH=os.pathsep.join([str(VERL), str(SOURCE), result.get("PYTHONPATH", "")]),
        SWE_SYNTHESIS_VERL_ROOT=str(VERL),
        LLM_API_URL=config["base_url"] + "/v1",
        LLM_API_KEY=result.get("TEACHER_API_KEY", "EMPTY"),
        LLM_MODEL=config["model"],
        SWE_AGENT_EXECUTION_BACKEND="remote",
        SWE_AGENT_HARNESS_PROFILE="claude_like",
        SWE_AGENT_EFFICIENCY_PROFILE="1",
        SWE_AGENT_TRAINING_SKIP_HIDDEN_REWARD="0",
        SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING="0",
        SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING="0",
        SWE_AGENT_TRAINING_PARTIAL_HIDDEN_TEST_REWARD="0",
        SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING="0",
        SWE_AGENT_REPEAT_REWARD_MODE="off",
        SWE_AGENT_TOOL_COMMAND_DIAGNOSTICS_ENABLED="1",
        SWE_AGENT_TOOL_COMMAND_FEEDBACK_ENABLED="1",
        SWE_AGENT_TOOL_COMMAND_SNAPSHOT_TIMEOUT_SECONDS="5",
        SWE_SFT_EXPORT_TOOL_CONTRACT_OUTCOMES="1",
        SWE_AGENT_TRAINING_TOOL_FAILURE_PENALTY_ENABLED="1",
        SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_ENABLED="0",
        SWE_AGENT_TRAINING_THINKING_REPEAT_PENALTY_ENABLED="0",
        SWE_AGENT_TRAINING_THINKING_REPEAT_AGGREGATE_ENABLED="0",
        SWE_AGENT_TRAINING_THINKING_REPEAT_CROSS_TURN_ENABLED="0",
        SWE_SFT_PYTHON=sys.executable,
    )
    return result


def request_json(url: str, *, body: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TEACHER_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TEACHER_API_KEY"]
    request = urllib.request.Request(url, headers=headers, data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=60 if body else 10) as response:
        return json.load(response)


def preflight(config: dict, *, generate: bool = False) -> None:
    missing = [
        name
        for name in ("torch", "ray", "pyarrow", "transformers", "omegaconf", "aiohttp", "tensordict")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ValueError("Python environment is missing: " + ", ".join(missing))
    paths = [config["source_parquet"], config["tokenizer"], config["tool_config"], *config["held_out"]]
    if config["input_parquet"]:
        paths.append(config["input_parquet"])
    for path in paths:
        if not Path(path).exists():
            raise ValueError(f"Required input does not exist: {path}")
    models = request_json(config["base_url"] + "/v1/models").get("data", [])
    model = next((row for row in models if row.get("id") == config["model"]), None)
    if model is None:
        raise ValueError(f"Teacher model {config['model']!r} is absent from /v1/models")
    context_length = model.get("max_model_len")
    if context_length and config["max_input_tokens"] + config["per_turn_tokens"] > int(context_length):
        raise ValueError(
            "TEACHER_MAX_INPUT_TOKENS + TEACHER_PER_TURN_TOKENS exceeds the teacher's "
            f"{context_length}-token context; reduce one of these settings"
        )
    print(
        json.dumps(
            {"teacher_model": model["id"], "checkpoint": model.get("root"), "max_model_len": model.get("max_model_len")}
        ),
        flush=True,
    )
    urls = [
        value.strip().rstrip("/")
        for value in os.environ.get("SWE_AGENT_EXECUTION_URLS", "").split(",")
        if value.strip()
    ]
    if not urls:
        raise ValueError("Set SWE_AGENT_EXECUTION_URLS to the running SWE execution services")

    def ready(url: str) -> tuple[str, bool]:
        try:
            with urllib.request.urlopen(url + "/ready", timeout=5) as response:
                return url, response.status == 200
        except OSError:
            return url, False

    results = list(ThreadPoolExecutor(max_workers=min(8, len(urls))).map(ready, urls))
    for url, healthy in results:
        print(json.dumps({"execution_url": url, "ready": healthy}), flush=True)
    if not any(healthy for _, healthy in results):
        raise ValueError("No ready SWE execution service")
    if generate:
        response = request_json(
            config["base_url"] + "/v1/chat/completions",
            body={
                "model": config["model"],
                "messages": [{"role": "user", "content": 'Return exactly {"ok": true}.'}],
                "max_tokens": 64,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print(
            json.dumps(
                {"smoke_response": response["choices"][0]["message"]["content"], "usage": response.get("usage")}
            ),
            flush=True,
        )


def run(command: list[object], env: dict) -> None:
    subprocess.run([str(value) for value in command], cwd=VERL, env=env, check=True)


def stage(action: str, root: Path, config: dict, env: dict, args: argparse.Namespace) -> None:
    python = [sys.executable, "-m"]
    selected = root / "mining/selected"
    synthesis = root / "synthesis"
    exported = root / "quality_source"
    quality = root / "quality"
    snapshot = root / "snapshot"
    mixed = root / "mixed"
    env = {
        **env,
        "SWE_SFT_OUTPUT_ROOT": str(quality),
        "SWE_SFT_SOURCE_MANIFEST": str(exported / "sources.json"),
        "SWE_SFT_RL_REFERENCE": config["source_parquet"],
        "SWE_SFT_JUDGE_URL": config["base_url"],
    }
    if action == "mine":
        if config["input_parquet"]:
            if not Path(config["input_parquet"]).is_file():
                raise ValueError(f"Teacher input does not exist: {config['input_parquet']}")
            print(f"Using explicit teacher input; skipping mining: {config['input_parquet']}", flush=True)
            return
        if (selected / "manifest.json").exists():
            print(f"Reusing frozen task selection: {selected}", flush=True)
            return
        run(
            [
                *python,
                "recipe.swe_agent.scripts.filter_training_tasks",
                "--source-parquet",
                config["source_parquet"],
                "--trajectory-records",
                config["rollout_dir"],
                "--history-records",
                config["rollout_dir"],
                "--min-step",
                config["min_step"],
                "--max-step",
                config["max_step"],
                "--require-evaluation-valid",
                "--require-no-history-success",
                "--exclude-tasks",
                *config["held_out"],
                "--min-attempts",
                config["min_attempts"],
                "--min-distinct-steps",
                config["min_distinct_steps"],
                "--max-teacher-tasks",
                config["max_tasks"],
                "--rollout-n",
                8,
                "--success-criterion",
                "hidden_pass",
                "--include-queue-failures",
                "--mode",
                "apply",
                "--output-dir",
                root / "mining",
                "--run-id",
                "selected",
            ],
            env,
        )
    elif action == "collect":
        run(
            [
                *python,
                "recipe.swe_trajectory_synthesis",
                "--input-parquet",
                config["input_parquet"] or selected / "teacher_tasks.parquet",
                "--output-dir",
                synthesis,
                "--tokenizer-path",
                config["tokenizer"],
                "--verl-root",
                VERL,
                "--tool-config",
                config["tool_config"],
                "--max-attempts",
                config["max_attempts"],
                "--concurrency",
                config["concurrency"],
                "--max-turns",
                config["max_turns"],
                "--response-length",
                config["response_length"],
                "--per-turn-max-tokens",
                config["per_turn_tokens"],
                "--max-input-tokens",
                config["max_input_tokens"],
                "--max-infrastructure-retries",
                config["max_infrastructure_retries"],
                "--operation-timeout",
                config["operation_timeout"],
                "--trajectory-timeout",
                config["trajectory_timeout"],
                "--progress-interval",
                config["progress_interval"],
                "--temperature",
                config["temperature"],
                "--seed",
                config["seed"],
                *(["--keep-going-after-success"] if config["keep_going_after_success"] else []),
                *(["--enable-thinking"] if config["enable_thinking"] else ["--no-thinking"]),
            ],
            env,
        )
    elif action == "export":
        summary_path = synthesis / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("terminal") != summary.get("tasks"):
            raise ValueError("Collection is incomplete; resume collect before freezing the quality source")
        if (exported / "export_report.json").exists():
            print(f"Reusing immutable teacher export: {exported}", flush=True)
            return
        run(
            [
                *python,
                "scripts.export_swe_teacher_rollouts",
                "--synthesis-dir",
                synthesis,
                "--output-dir",
                exported,
                "--tokenizer-path",
                config["tokenizer"],
            ],
            env,
        )
    elif action in {"prepare", "pilot", "filter", "check"}:
        if (
            action == "filter"
            and (quality / "full/statistics.json").exists()
            and (quality / "full/core_train.parquet").exists()
        ):
            status = json.loads((quality / "full/run_status.json").read_text())
            if status.get("status") == "filter_complete":
                print(f"Reusing completed quality filter: {quality / 'full'}", flush=True)
                return
        command = ["bash", VERL / "scripts/run_swe_sft_quality.sh", "full" if action == "filter" else action]
        if action == "check":
            if not args.labels:
                raise ValueError("check requires --labels PATH")
            command.append(args.labels)
        else:
            command += [
                "--judge-model", config["model"], "--judge-concurrency", 2, "--case-concurrency", 2,
                "--max-response-tokens", config["response_length"],
                "--max-tool-dispatches", config["max_turns"],
                "--relaxed-max-tool-dispatches", config["max_turns"],
            ]
            if action == "filter" and args.allow_unaudited_pilot:
                command.append("--allow-unaudited-pilot")
        run(command, env)
    elif action == "snapshot":
        if (snapshot / "snapshot_report.json").exists():
            print(f"Reusing immutable SFT snapshot: {snapshot}", flush=True)
            return
        run(
            [
                *python,
                "scripts.export_swe_sft_snapshot",
                "--run-dir",
                quality / "full",
                "--output-dir",
                snapshot,
                "--rl-reference-parquet",
                config["source_parquet"],
            ],
            env,
        )
    elif action == "mix":
        if (mixed / "mixture_report.json").exists():
            print(f"Reusing immutable mixed dataset: {mixed}", flush=True)
            return
        command = [
            *python,
            "scripts.mix_swe_sft_data",
            "--teacher-parquet",
            snapshot / "core_train.parquet",
            "--teacher-fraction",
            config["teacher_fraction"],
            "--val-fraction",
            config["val_fraction"],
            "--tokenizer-path",
            config["tokenizer"],
            "--max-length",
            config["max_length"],
            "--output-dir",
            mixed,
            "--seed",
            config["seed"],
        ]
        if config["teacher_fraction"] < 1:
            command += ["--existing-parquet", config["existing_train"]]
        for path in config["held_out"]:
            command += ["--exclude-task-parquet", path]
        run(command, env)
    elif action == "train":
        env.update(
            SFT_TRAIN_FILE=str(mixed / "train.parquet"),
            SFT_TEACHER_VAL_FILE=str(mixed / "val.parquet"),
            SFT_DATA_MANIFEST=str(mixed / "mixture_report.json"),
            MODEL_PATH=setting("MODEL_PATH", config["tokenizer"]),
            EXPERIMENT_NAME=setting("EXPERIMENT_NAME", "swe_teacher397_hard_sft"),
            SFT_BATCH_SIZE=setting("SFT_BATCH_SIZE", 8),
            SFT_LR=setting("SFT_LR", "5e-7"),
            SFT_EPOCHS=setting("SFT_EPOCHS", 1),
            SFT_MAX_LENGTH=str(config["max_length"]),
            SFT_TEST_FREQ=setting("SFT_TEST_FREQ", 2),
            SFT_SAVE_FREQ=setting("SFT_SAVE_FREQ", 2),
            SFT_TEACHER_TEST_FREQ=setting("SFT_TEACHER_TEST_FREQ", 2),
            MAX_RESPONSE_LENGTH=setting("MAX_RESPONSE_LENGTH", 98304),
            PPO_MAX_TOKEN_LEN_PER_GPU=setting("PPO_MAX_TOKEN_LEN_PER_GPU", 32768),
            ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=setting("ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU", 32768),
            CP=setting("CP", 4),
            SWE_AGENT_MAX_TURNS=setting("SWE_AGENT_MAX_TURNS", 150),
            VAL_FILES=setting("VAL_FILES", json.dumps(config["held_out"][:1])),
            VALIDATION_DATA_DIR=setting("VALIDATION_DATA_DIR", root / "student_validation"),
        )
        env.update(SFT_VAL_BEFORE_TRAIN="True", SFT_TEACHER_VAL_BEFORE_TRAIN="True")
        run(
            [
                "bash",
                VERL / "scripts/run_swe_agent_sft_qwen3_5_35b_a3b.sh",
                "trainer.resume_mode=disable",
                "+data.apply_chat_template_kwargs.enable_thinking=True",
                "trainer.validation_data_dir=" + env["VALIDATION_DATA_DIR"],
            ],
            env,
        )
    else:
        raise ValueError(f"Unknown stage {action}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "mine",
            "collect",
            "export",
            "prepare",
            "pilot",
            "check",
            "filter",
            "snapshot",
            "mix",
            "all",
            "train",
        ],
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(setting("TEACHER_OUTPUT_ROOT", VERL / "analysis/swe_teacher_sft/v1"))
    )
    parser.add_argument(
        "--allow-unaudited-pilot",
        action="store_true",
        help="Explicitly use the existing unaudited full-filter override",
    )
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--generate-smoke", action="store_true", help="preflight: also generate a tiny response")
    args = parser.parse_args()
    config = configuration()
    env = environment(config)
    if args.action == "preflight":
        preflight(config, generate=args.generate_smoke)
        return
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = root / "pipeline_config.json"
    if contract.exists() and resume_configuration(json.loads(contract.read_text())) != resume_configuration(config):
        raise ValueError("Pipeline settings changed; choose a new TEACHER_OUTPUT_ROOT")
    if not contract.exists():
        with contract.open("x") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
    for action in ["mine", "collect", "export", "filter", "snapshot", "mix"] if args.action == "all" else [args.action]:
        print(f"Teacher SFT stage: {action}", flush=True)
        stage(action, root, config, env, args)


if __name__ == "__main__":
    main()
