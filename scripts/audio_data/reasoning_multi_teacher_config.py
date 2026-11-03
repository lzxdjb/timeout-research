"""Build configuration for the opt-in SWE/audio/general reasoning profile."""

import argparse
import json
import os
from pathlib import Path

from multi_teacher_config import hydra_literal

TRAIN_NAMES = ("openmath_29", "math_mixture_42", "arxivqa_136", "deepvision_227")
VAL_NAMES = ("aime_2024", "aime_2025", "gpqa_diamond", "mmlu_pro", "bbh")
GENERAL_SOURCES = ("reasoning_openmath", "reasoning_math_mixture", "reasoning_arxivqa", "reasoning_deepvision")


def train_files(env):
    source = Path(env["SWE_SOURCE"])
    root = Path(env["GENERAL_REASONING_DATA_ROOT"])
    audio = Path(env.get("AUDIO_ROOT", str(source / "data/audio_opd_pilot_v1")))
    return [
        str(source / "data/swe_benchmarks/verl/swe_rebench_v2_reference_verified_snapshot_2000.parquet"),
        str(audio / "train.opd.parquet"),
        *(str(root / "frontier" / f"{name}.parquet") for name in TRAIN_NAMES),
    ]


def default_mix():
    phases = ((20, 12, 14, 13, 9), (60, 24, 9, 9, 6), (20, 36, 5, 4, 3))
    return [
        {
            "steps": steps,
            "quotas": {
                "swe": swe,
                "audio_librispeech": libri,
                "audio_aishell1": aishell,
                "audio_clotho_aqa": clotho,
                **dict.fromkeys(GENERAL_SOURCES, 4),
            },
        }
        for steps, swe, libri, aishell, clotho in phases
    ]


def validation_config(env):
    root = Path(env["GENERAL_REASONING_DATA_ROOT"])
    files = (
        json.loads(env["GENERAL_REASONING_VAL_FILES"])
        if env.get("GENERAL_REASONING_VAL_FILES")
        else [str(root / "reasoning-benchmarks/verl" / f"{name}.parquet") for name in VAL_NAMES]
    )
    if not isinstance(files, list) or not files or any(not isinstance(path, str) or not path for path in files):
        raise ValueError("GENERAL_REASONING_VAL_FILES must be a nonempty JSON list of paths")
    max_samples = json.loads(env.get("GENERAL_REASONING_MAX_SAMPLES", "-1"))
    if max_samples is not None and (type(max_samples) is not int or max_samples == 0 or max_samples < -1):
        raise ValueError("GENERAL_REASONING_MAX_SAMPLES must be null, -1, or a positive integer")
    prompt = int(env.get("max_prompt_length", env.get("MAX_PROMPT_LENGTH", "8192")))
    response = int(env.get("max_response_length", env.get("MAX_RESPONSE_LENGTH", "253951")))
    context = int(env.get("TEACHER_MAX_MODEL_LEN", "262144"))
    concurrency = int(env.get("GENERAL_REASONING_CONCURRENCY", "32"))
    timeout = float(env.get("GENERAL_REASONING_TIMEOUT", "3600"))
    if min(prompt, response, context, concurrency, timeout) <= 0 or prompt + response + 1 > context:
        raise ValueError("Reasoning limits must be positive and reserve one token in the teacher context")
    return {
        "enabled": True,
        "val_files": files,
        "max_samples": max_samples,
        "concurrency": concurrency,
        "output_dir": env["GENERAL_REASONING_OUTPUT_DIR"],
        "max_prompt_length": prompt,
        "response_length": response,
        "max_model_len": context,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "timeout": timeout,
        "failure_policy": "error",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("train", "mix", "validation"))
    args = parser.parse_args()
    if args.component == "train":
        # The existing Megatron wrapper expands train_path into a Bash array.
        # Avoid separator whitespace so this list stays one Hydra argument.
        print(json.dumps(train_files(os.environ), separators=(",", ":")))
    elif args.component == "mix":
        print(json.dumps(default_mix(), separators=(",", ":")))
    else:
        print(hydra_literal(validation_config(os.environ)))


if __name__ == "__main__":
    main()
