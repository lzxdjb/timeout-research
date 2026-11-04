"""Regressions for BBH representation and final-answer selection failures."""

import pytest

from verl.trainer.ppo.bbh_validation_scoring import score_bbh_validation

QUESTION = "Which option is correct?\nOptions:\n(A) first\n(B) second\n(C) third"


@pytest.mark.parametrize("answer", ["B", "(B)", r"\boxed{B}", r"\boxed{(B)}", "So the answer is (B)."])
def test_choice_answer_representation_and_box_protocol_are_separate(answer):
    result = score_bbh_validation(answer, "B", QUESTION)
    assert result["answer_acc"] == result["representation_acc"] == 1.0
    assert result["boxed_answer_acc"] == float("boxed" in answer)


@pytest.mark.parametrize("label", ["K", "M", "R", "Z"])
def test_old_exact_metadata_for_later_option_labels_is_compatible(label):
    prompt = f"Options:\n(A) first\n({label}) last"
    result = score_bbh_validation(r"\boxed{" + label + "}", f"({label})", prompt, {"answer_type": "exact"})
    assert result["answer_acc"] == 1.0
    assert result["answer_type"] == "multiple_choice"
    assert result["option_labels"] == ["A", label]


@pytest.mark.parametrize(
    "solution,expected_source",
    [
        (r"The answer is B. Finally, \boxed{(C)}", "boxed"),
        (r"The answer is B. Finally, \boxed{Z}", "boxed"),
        (r"\boxed{B} but the final answer is C.", "explicit"),
        (r"\boxed{B} then \boxed{}", "boxed"),
        (r"\boxed{B} then \boxed{C", "boxed"),
        (r"The answer is B. Finally, \boxed C", "boxed"),
    ],
)
def test_wrong_or_unusable_final_answer_never_recovers_an_earlier_correct_one(solution, expected_source):
    result = score_bbh_validation(solution, "B", QUESTION)
    assert result["answer_acc"] == result["representation_acc"] == 0.0
    assert result["answer_source"] == expected_source


def test_invalid_actual_option_is_not_accepted_from_earlier_thinking():
    result = score_bbh_validation("The answer is B.\nFinal answer: Z", "B", QUESTION)
    assert result["answer_acc"] == 0.0
    assert result["answer"] == "Z"


def test_nested_box_and_markers_inside_box_cannot_supply_a_choice():
    for solution in (r"\boxed{A \boxed{B}}", r"The answer is B. \boxed{A: the answer is B}"):
        result = score_bbh_validation(solution, "B", QUESTION)
        assert result["answer_acc"] == result["representation_acc"] == 0.0
        assert result["answer_source"] == "boxed"
    assert score_bbh_validation(r"\boxed{A \boxed{B}}", "B", QUESTION)["parse_error"] == "nested_final_box"
    # A later independent final answer still supersedes an invalid earlier box.
    assert score_bbh_validation(r"\boxed{A \boxed{C}} \boxed{B}", "B", QUESTION)["answer_acc"] == 1.0


def test_thinking_answer_is_not_a_final_submission():
    result = score_bbh_validation(r"<think>The answer is B. \boxed{B}</think>", "B", QUESTION)
    assert result["answer_acc"] == 0.0
    result = score_bbh_validation(r"<think>\boxed{B}", "B", QUESTION)
    assert result["answer_acc"] == 0.0
    assert result["parse_error"] == "unfinished_thinking"
    result = score_bbh_validation(r"<think>\boxed{C}</think>So the answer is (B).", "B", QUESTION)
    assert result["answer_acc"] == 1.0


def test_options_are_taken_from_question_not_system_protocol():
    prompt = [
        {"role": "system", "content": "Example choices:\n(A) first\n(B) second"},
        {"role": "user", "content": "Return the literal string (B)."},
    ]
    result = score_bbh_validation("B", "(B)", prompt, {"answer_type": "multiple_choice"})
    assert result["answer_acc"] == 0.0
    assert result["option_labels"] == []
    assert result["answer_type"] == "exact"


@pytest.mark.parametrize("prediction", [r"\boxed{apple, banana, cherry}", r"\boxed{\text{apple banana cherry}}"])
def test_word_sorting_separator_and_latex_cleanup_is_only_diagnostic(prediction):
    result = score_bbh_validation(
        prediction, "apple banana cherry", "Sort the words.", {"meta_json": '{"task":"word_sorting"}'}
    )
    assert result["answer_acc"] == 0.0
    assert result["representation_acc"] == result["boxed_representation_acc"] == 1.0
    assert result["task"] == "word_sorting"
    assert score_bbh_validation(
        r"\boxed{banana, apple, cherry}", "apple banana cherry", "Sort.", {"task": "word_sorting"}
    )["representation_acc"] == 0.0


def test_boolean_latex_cleanup_is_only_diagnostic_and_task_specific():
    result = score_bbh_validation(r"\boxed{\text{True}}", "True", "Evaluate.", {"task": "boolean_expressions"})
    assert result["answer_acc"] == 0.0
    assert result["representation_acc"] == 1.0
    assert result["diagnostic_reason"] == "boolean_text_wrapper"
    assert score_bbh_validation(r"\boxed{\text{True}}", "True", "Return exact text.")["representation_acc"] == 0.0


def test_choice_latex_cleanup_stays_separate_from_primary_score():
    result = score_bbh_validation(r"\boxed{\text{(B)}}", "B", QUESTION)
    assert result["answer_acc"] == 0.0
    assert result["representation_acc"] == 1.0


def test_dyck_symbols_remain_exact_in_both_scores():
    assert score_bbh_validation(r"\boxed{])>}", "])>", "Complete.", {"task": "dyck_languages"})["answer_acc"] == 1.0
    for prediction in (r"\boxed{)>]}", r"\boxed{>}", r"\boxed{[]()>}", r"\boxed{\text{])>}}"):
        result = score_bbh_validation(prediction, "])>", "Complete.", {"task": "dyck_languages"})
        assert result["answer_acc"] == result["representation_acc"] == 0.0


def test_exact_answer_retains_punctuation_in_boxed_payload():
    result = score_bbh_validation(r"\boxed{a,b}", "a b", "Return exactly.")
    assert result["answer_acc"] == result["representation_acc"] == 0.0


def test_empty_ground_truth_and_empty_response_never_match():
    assert score_bbh_validation("", "", "Question")["answer_acc"] == 0.0
