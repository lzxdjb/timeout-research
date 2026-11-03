"""Compatibility boundaries for the opt-in general reasoning profile."""

import copy

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

from verl.utils.dataset.audio_opd_dataset import route_audio_opd_row
from verl.utils.dataset.reasoning_opd_dataset import ReasoningOPDMixedDataset, route_reasoning_opd_row
from verl.utils.reward_score import swe_audio_opd, swe_audio_reasoning_opd


@pytest.mark.parametrize("source", ["122b_openmath", "122b_math_mixture", "122b_arxivqa", "122b_deepvision"])
def test_general_routing_preserves_original_prompt_images_references(source):
    row = {
        "data_source": source,
        "agent_name": "general_reasoning_122b_test",
        "prompt": [{"role": "user", "content": "<image> solve this"}],
        "images": [{"bytes": b"image-data"}],
        "reward_model": {"ground_truth": "B"},
    }
    original = copy.deepcopy(row)
    fields = route_reasoning_opd_row(row)
    assert row == original
    assert fields == {
        "teacher_route": "swe",
        "agent_name": "general_reasoning_agent",
        "domain": "general",
        "opd_source": "reasoning_" + source.removeprefix("122b_"),
        "metric_data_source": source,
        "is_general_reasoning": True,
    }
    assert not {"prompt", "images", "reward_model", "data_source"} & fields.keys()


@pytest.mark.parametrize(
    "row",
    [
        {"data_source": "swe_agent_verl", "agent_name": "swe_tool_agent"},
        {"data_source": "benchmark_swe", "agent_name": "swe_tool_agent"},
        {"data_source": "audio_librispeech", "agent_name": "single_turn_agent", "audios": [{"audio": "/a.wav"}]},
    ],
)
def test_existing_domains_use_the_exact_existing_router(row):
    assert route_reasoning_opd_row(row) == {**route_audio_opd_row(row), "is_general_reasoning": False}


@pytest.mark.parametrize("source", ["benchmark_hle", "122b_unknown"])
def test_unsupported_reasoning_cannot_silently_become_swe(source):
    with pytest.raises(ValueError, match="Unsupported general"):
        route_reasoning_opd_row({"data_source": source, "agent_name": "general_reasoning_122b_test"})


def test_mixed_dataset_keeps_legacy_general_agent_out_of_swe_router(tmp_path):
    rows = [
        {
            "data_source": "swe_agent_verl",
            "agent_name": "swe_tool_agent",
            "prompt": [{"role": "user", "content": "repair"}],
        },
        {
            "data_source": "122b_openmath",
            "agent_name": "general_reasoning_122b_test",
            "prompt": [{"role": "user", "content": "reason"}],
        },
    ]
    path = tmp_path / "mixed.parquet"
    Dataset.from_list(rows).to_parquet(path)
    dataset = ReasoningOPDMixedDataset(
        [str(path)], tokenizer=None, config=OmegaConf.create({"filter_overlong_prompts": False})
    )
    assert list(dataset.dataframe["is_general_reasoning"]) == [False, True]
    assert list(dataset.dataframe["teacher_route"]) == ["swe", "swe"]
    assert dataset[1]["raw_prompt"] == rows[1]["prompt"]


@pytest.mark.parametrize(
    "solution,score,raw_acc",
    [
        (r"The answer is \boxed{B}", 1.0, 1.0),
        (r"The answer is \boxed{C}", 0.0, 0.0),
        ("The answer is B", 0.0, 1.0),
        (r"\boxed{}", 0.0, 0.0),
        ("", 0.0, 0.0),
    ],
)
def test_real_deterministic_reward_is_binary_with_no_format_penalty(solution, score, raw_acc):
    result = swe_audio_reasoning_opd.compute_general_score(
        "benchmark_gpqa_diamond",
        solution,
        "B",
        {"answer_type": "multiple_choice"},
        llm_judge_enabled=True,
        format_penalty=9.0,
    )
    assert result["score"] == result["acc"] == result["final_score"] == score
    assert result["deterministic_acc"] == raw_acc
    assert result["format_reward"] == 0.0
    assert result["llm_judge_used"] == 0.0


@pytest.mark.parametrize("source", ["swe_agent_verl", "audio_librispeech"])
def test_other_reward_calls_delegate_unchanged(monkeypatch, source):
    expected = object()
    calls = []
    monkeypatch.setattr(
        swe_audio_opd, "compute_score", lambda *args, **kwargs: (calls.append((args, kwargs)), expected)[1]
    )
    info = {"same": True}
    actual = swe_audio_reasoning_opd.compute_score(source, "response", "reference", info, same_option=2)
    assert actual is expected
    assert calls == [((source, "response", "reference", info), {"same_option": 2})]
