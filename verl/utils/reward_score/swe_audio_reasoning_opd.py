"""Binary general validation scoring, with unchanged SWE/audio reward dispatch.

Only the synchronous deterministic scorer is loaded. Neither the optional LLM
judge nor a learned reward model is used by this profile.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

_GENERAL_SOURCES = frozenset(
    {
        "122b_openmath",
        "122b_math_mixture",
        "122b_arxivqa",
        "122b_deepvision",
        "benchmark_aime_2024",
        "benchmark_aime_2025",
        "benchmark_gpqa_diamond",
        "benchmark_mmlu_pro",
        "benchmark_bbh",
    }
)


def _default_reward_path() -> Path:
    return Path(__file__).resolve().parents[5] / "stock-rl-reflect/recipe/122B-test/rl122b_test/rewarding.py"


@lru_cache(maxsize=4)
def _load_deterministic_scorer(path: str):
    reward_path = Path(path).expanduser().resolve()
    # The recipe entry point re-exports this module and changes sys.path. Load
    # its deterministic implementation directly to keep other profiles isolated.
    if reward_path.name == "reward_122b_test.py":
        reward_path = reward_path.parent / "rl122b_test/rewarding.py"
    if not reward_path.is_file():
        raise FileNotFoundError(f"General reasoning deterministic scorer not found: {reward_path}")
    spec = importlib.util.spec_from_file_location("_verl_general_reasoning_deterministic_reward", reward_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load deterministic reasoning scorer: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def compute_general_score(
    data_source, solution_str="", ground_truth=None, extra_info=None, *, reasoning_reward_path=None, **kwargs
):
    """Correct boxed answers earn one; all other answers earn zero."""
    if data_source not in _GENERAL_SOURCES:
        raise ValueError(f"Unsupported general reasoning reward source: {data_source!r}")
    path = reasoning_reward_path or os.environ.get("GENERAL_REASONING_REWARD_PATH") or _default_reward_path()
    scorer = _load_deterministic_scorer(str(path))
    # Caller-provided judge/format options cannot enable a judge or penalties.
    result = scorer(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        return_dict=True,
        format_penalty=0.0,
    )
    result = dict(result)
    score = float(bool(result.get("format_valid")) and float(result.get("deterministic_acc", 0.0)) >= 1.0)
    result.update(
        score=score,
        acc=score,
        deterministic_score=score,
        final_acc=score,
        final_score=score,
        format_reward=0.0,
        llm_judge_used=0.0,
        llm_judge_skip_reason="disabled_in_general_reasoning_profile",
    )
    return result


def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    if data_source in _GENERAL_SOURCES:
        return compute_general_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    from verl.utils.reward_score.swe_audio_opd import compute_score as swe_audio_score

    return swe_audio_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
