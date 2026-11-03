"""Opt-in mixed reward dispatch; SWE scoring delegates to the existing recipe."""


def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    if isinstance(data_source, str) and data_source.startswith("audio_"):
        from verl.utils.reward_score.audio import compute_score as audio_score

        result = audio_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
        return {"score": float(result), "data_source": data_source}
    from recipe.swe_agent.reward_function import compute_score as swe_score

    return swe_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
