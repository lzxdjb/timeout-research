"""Build literal Hydra values for the opt-in frozen-audio OPD launcher."""

import argparse
import json
import os
import re


def hydra_literal(value):
    if isinstance(value, dict):
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in value):
            raise ValueError("OPD config keys must be identifiers")
        return "{" + ",".join(f"{key}:{hydra_literal(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(item) for item in value) + "]"
    return json.dumps(value, allow_nan=False)


def teacher_config(route, env):
    audio = route == "audio"
    prefix = "AUDIO_TEACHER_" if audio else "TEACHER_"
    limit = int(env[prefix + "MAX_MODEL_LEN"])
    prompt = int(env.get("AUDIO_MAX_PROMPT_LENGTH", "8192") if audio else env["MAX_PROMPT_LENGTH"])
    response = int(env.get("AUDIO_MAX_RESPONSE_LENGTH", "2048") if audio else env["MAX_RESPONSE_LENGTH"])
    external = {
        "_target_": "verl.workers.config.ExternalTeacherConfig",
        "base_url": env[prefix + "BASE_URL"],
        "model_name": env[prefix + "MODEL_NAME"],
        "max_model_len": limit,
        "max_concurrent_requests": int(env[prefix + "MAX_CONCURRENT_REQUESTS"]),
        "request_timeout_s": float(env.get(prefix + "REQUEST_TIMEOUT_S", "1800")),
        "max_retries": int(env.get(prefix + "MAX_RETRIES", "2")),
        "retry_backoff_s": float(env.get(prefix + "RETRY_BACKOFF_S", "2")),
        "protocol": "audio_candidate_score" if audio else "text_token_ids",
    }
    if audio:
        external.update(
            tokenizer_path=env["AUDIO_TEACHER_TOKENIZER_PATH"],
            audio_prompt_length=prompt,
            audio_response_length=response,
        )
    else:
        external["unsupported_token_ids"] = list(range(248070, 248077))
    return {
        "_target_": "verl.workers.config.DistillationTeacherModelConfig",
        "key": route,
        "backend": "external",
        "num_replicas": 0,
        "external": external,
        "inference": {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "prompt_length": prompt,
            "response_length": response,
            "max_model_len": limit,
            "tensor_model_parallel_size": 1,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("swe", "audio", "mix"))
    args = parser.parse_args()
    if args.component == "mix":
        value = json.loads(os.environ["OPD_MIX_SCHEDULE"])
        if not isinstance(value, list) or not value:
            raise ValueError("OPD_MIX_SCHEDULE must be a nonempty JSON list")
        steps = sum(phase["steps"] for phase in value)
        if steps < int(os.environ.get("TOTAL_TRAINING_STEPS", "100")):
            raise ValueError("OPD_MIX_SCHEDULE does not cover TOTAL_TRAINING_STEPS")
        batch = int(os.environ.get("TRAIN_BATCH_SIZE", "64"))
        for phase in value:
            if sum(phase["quotas"].values()) != batch:
                raise ValueError("Each phase's quotas must sum to TRAIN_BATCH_SIZE")
    else:
        value = teacher_config(args.component, os.environ)
    print(hydra_literal(value))


if __name__ == "__main__":
    main()
