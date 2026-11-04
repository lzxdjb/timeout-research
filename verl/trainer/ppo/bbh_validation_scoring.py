"""Deterministic, validation-only BBH final-answer scoring.

``answer_acc`` accepts ordinary final answers as well as boxed answers and
normalizes multiple-choice labels against options actually present in the
question. ``representation_acc`` additionally accepts comma-separated word
sorting and outer LaTeX text wrappers. It is a diagnostic, not an official BBH
score. All accuracy values are binary fractions (0 or 1), not percentages.

This module intentionally does not import or change the shared training reward.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

PARSER_VERSION = "bbh_validation_v1"
_BOX_COMMAND = re.compile(r"\\boxed\b")
_ANSWER_MARKER = re.compile(r"\b(?:final\s+answer|(?:the\s+)?answer)\s*(?:is\b\s*|[:：]\s*)", re.IGNORECASE)
_OPTION = re.compile(r"^\s*\(([A-Z])\)\s*\S", re.MULTILINE)
_CHOICE = re.compile(r"(?:([A-Z])|\(([A-Z])\))[.]?", re.IGNORECASE)
_TEXT_WRAPPER = re.compile(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", re.DOTALL)


def _question_text(prompt: str | list[dict]) -> str:
    if isinstance(prompt, str):
        return prompt
    # System prompts describe formatting; only user questions define options.
    return "\n".join(str(message.get("content") or "") for message in prompt if message.get("role") == "user")


def _task_name(extra_info: Mapping[str, Any] | None) -> str:
    info = extra_info or {}
    if info.get("task"):
        return str(info["task"])
    meta = info.get("meta_json") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (TypeError, ValueError):
            return ""
    return str(meta.get("task") or "") if isinstance(meta, Mapping) else ""


def _is_escaped(value: str, position: int) -> bool:
    start = position
    while start > 0 and value[start - 1] == "\\":
        start -= 1
    return (position - start) % 2 == 1


def _boxed_events(text: str) -> list[tuple[int, int, str, str]]:
    """Collect outer box spans; nested boxes cannot become separate answers."""
    events = []
    covered_until = 0
    for command in _BOX_COMMAND.finditer(text):
        if command.start() < covered_until:
            continue
        opening = command.end()
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening == len(text) or text[opening] != "{":
            events.append((command.start(), command.end(), "", "malformed_final_box"))
            continue
        depth = 1
        end = len(text)
        answer, error = "", "malformed_final_box"
        for position in range(opening + 1, len(text)):
            char = text[position]
            if char == "{" and not _is_escaped(text, position):
                depth += 1
            elif char == "}" and not _is_escaped(text, position):
                depth -= 1
                if depth == 0:
                    end = position + 1
                    answer = text[opening + 1 : position].strip()
                    error = "" if answer else "empty_final_box"
                    if _BOX_COMMAND.search(answer):
                        answer, error = "", "nested_final_box"
                    break
        events.append((command.start(), end, answer, error))
        covered_until = end
    return events


def _final_answer(solution: str) -> tuple[str, str, str]:
    """Return the last final-answer event; never retry an older answer.

    A later explicit answer supersedes an earlier box. A later malformed or
    unusable box stays authoritative instead of exposing a correct earlier box
    or an answer mentioned in reasoning. An unfinished thinking section has no
    final answer.
    """
    text = str(solution or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return "", "none", "unfinished_thinking"
    boxes = _boxed_events(text)
    markers = []
    box_index = 0
    for marker in _ANSWER_MARKER.finditer(text):
        while box_index < len(boxes) and boxes[box_index][1] <= marker.start():
            box_index += 1
        if box_index == len(boxes) or marker.start() < boxes[box_index][0]:
            markers.append(marker)
    last_box = boxes[-1] if boxes else None
    last_marker = markers[-1] if markers else None
    if last_box and (not last_marker or last_box[0] > last_marker.start()):
        return last_box[2], "boxed", last_box[3]
    if last_marker:
        suffix = text[last_marker.end() :].strip()
        answer = suffix.splitlines()[0].strip() if suffix else ""
        # A sentence-ending period belongs to the explicit answer phrase.
        return answer.removesuffix(".").strip(), "explicit", "" if answer else "empty_final_answer"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1], "last_line", "") if lines else ("", "none", "missing_final_answer")


def _exact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _unwrap_text(value: str) -> str:
    value = value.strip()
    while match := _TEXT_WRAPPER.fullmatch(value):
        value = match[1].strip()
    return value


def _choice_label(value: str, labels: set[str]) -> str:
    match = _CHOICE.fullmatch(value.strip())
    label = (match[1] or match[2]).upper() if match else ""
    return label if label in labels else ""


def score_bbh_validation(
    solution: str,
    ground_truth: Any,
    prompt: str | list[dict],
    extra_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the final BBH answer without requiring the training box protocol.

    ``answer_acc`` preserves exact-match case/whitespace normalization for
    non-choice tasks. Parentheses are optional for choices, but the selected
    label must appear among the question's A-Z options. Old parquet metadata
    marking K-Z targets as ``exact`` therefore remains usable without rewriting
    it. Optional representation changes are isolated in ``representation_acc``.
    """
    answer, source, parse_error = _final_answer(solution)
    labels = set(_OPTION.findall(_question_text(prompt)))
    # A single parenthesized letter in prose is not a multiple-choice roster.
    if len(labels) < 2:
        labels = set()
    target = "" if ground_truth is None else str(ground_truth).strip()
    target_label = _choice_label(target, labels)
    task = _task_name(extra_info)
    answer_type = "multiple_choice" if target_label else "exact"
    diagnostic_reason = ""
    if target_label:
        prediction = _choice_label(answer, labels)
        correct = bool(prediction) and prediction == target_label
        representation_prediction = _choice_label(_unwrap_text(answer), labels)
        representation_correct = bool(representation_prediction) and representation_prediction == target_label
        if representation_correct and not correct:
            diagnostic_reason = "choice_text_wrapper"
    else:
        correct = bool(answer and target) and _exact(answer) == _exact(target)
        representation_correct = correct
        if task == "word_sorting" and answer and target:
            words = re.split(r"[\s,]+", _unwrap_text(answer).strip().casefold())
            representation_correct = words == target.casefold().split()
            if representation_correct and not correct:
                diagnostic_reason = "word_sorting_representation"
        elif task == "boolean_expressions" and answer and target:
            boolean = _unwrap_text(answer).casefold()
            representation_correct = boolean in {"true", "false"} and boolean == target.casefold()
            if representation_correct and not correct:
                diagnostic_reason = "boolean_text_wrapper"
    has_boxed_answer = source == "boxed" and bool(answer) and not parse_error
    return {
        "answer_acc": float(correct),
        "representation_acc": float(representation_correct),
        "boxed_answer_acc": float(has_boxed_answer and correct),
        "boxed_representation_acc": float(has_boxed_answer and representation_correct),
        "answer": answer,
        "answer_source": source,
        "answer_type": answer_type,
        "option_labels": sorted(labels),
        "task": task,
        "parser_version": PARSER_VERSION,
        "has_boxed_answer": has_boxed_answer,
        "parse_error": parse_error,
        "diagnostic_reason": diagnostic_reason,
    }
