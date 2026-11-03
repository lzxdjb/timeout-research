"""Explicit rewards for audio records exported by scripts.audio_data.convert_verl.

Load this file with verl's custom reward-function hook. Open-ended answers and
incompletely annotated multi-label targets intentionally have no rule reward.
"""

import json
import re
import unicodedata


def normalize_text(text, language="en"):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    if language.lower().startswith(("zh", "cmn")):
        return "".join(text.split())
    return " ".join(text.split())


def error_rate(prediction, reference, language="en"):
    """Mandarin CER or space-delimited WER; no translation/number rewriting."""
    hypothesis, target = normalize_text(prediction, language), normalize_text(reference, language)
    chinese = language.lower().startswith(("zh", "cmn"))
    hypothesis = list(hypothesis) if chinese else hypothesis.split()
    target = list(target) if chinese else target.split()
    previous = list(range(len(hypothesis) + 1))
    for i, token in enumerate(target, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1] / max(1, len(target))


def _answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text or "</think>" in text:
        return ""
    tagged = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if tagged:
        return tagged[-1].strip()
    return text


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return [0,1]; invalid reward specifications raise instead of training on guesses."""
    specification = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(specification, dict):
        raise ValueError("Audio ground_truth must be the JSON reward specification from convert_verl")
    task = specification.get("task")
    if task not in {"asr", "closed_qa", "music_family", "classification"}:
        raise ValueError(f"No deterministic audio reward for task {task!r}")
    reference = specification.get("reference")
    references = [reference] if isinstance(reference, str) else reference
    if not isinstance(references, list) or not references or not all(isinstance(x, str) and x for x in references):
        raise ValueError("Audio reward requires nonempty string references")
    if task == "classification" and specification.get("label_set_complete") is not True:
        raise ValueError("Incomplete positive-label sets cannot be scored as complete classification targets")
    aliases = specification.get("reference_aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
        raise ValueError("reference_aliases must be a list of strings")
    language = specification.get("language", "en")
    if not isinstance(language, str) or not language:
        raise ValueError("Audio reward requires an explicit language string")
    if any(not normalize_text(value, language) for value in references + aliases):
        raise ValueError("References and aliases must be nonempty after normalization")
    if not isinstance(solution_str, str) or len(solution_str) > 20000:
        return 0.0
    answer = _answer(solution_str)
    if not normalize_text(answer, language):
        return 0.0
    if task == "asr":
        return max(0.0, 1.0 - min(error_rate(answer, target, language) for target in references + aliases))
    if task == "classification":
        try:
            predicted = json.loads(answer)
        except (ValueError, TypeError):
            return 0.0
        if (
            not isinstance(predicted, list)
            or not predicted
            or not all(isinstance(x, str) and normalize_text(x, language) for x in predicted)
        ):
            return 0.0
        predicted = {normalize_text(x, language) for x in predicted}
        target = {normalize_text(x, language) for x in references}
        return 2.0 * len(predicted & target) / (len(predicted) + len(target))
    return float(normalize_text(answer, language) in {normalize_text(x, language) for x in references + aliases})
