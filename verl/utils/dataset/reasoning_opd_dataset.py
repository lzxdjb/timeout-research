"""Routing used exclusively by the opt-in SWE/audio/general OPD profile."""

from verl.utils.dataset.audio_opd_dataset import AudioOPDMixedDataset, route_audio_opd_row
from verl.utils.dataset.rl_dataset import RLHFDataset

GENERAL_TRAIN_SOURCES = frozenset({"122b_openmath", "122b_math_mixture", "122b_arxivqa", "122b_deepvision"})
GENERAL_VALIDATION_SOURCES = frozenset(
    f"benchmark_{name}" for name in ("aime_2024", "aime_2025", "gpqa_diamond", "mmlu_pro", "bbh")
)


def route_reasoning_opd_row(row):
    """Keep existing SWE/audio routing and give selected reasoning rows their own loop."""
    source = row.get("data_source")
    if source not in GENERAL_TRAIN_SOURCES | GENERAL_VALIDATION_SOURCES:
        # An unsupported reasoning source must not silently become a SWE task.
        if (isinstance(source, str) and source.startswith("122b_")) or row.get("agent_name") in {
            "general_reasoning_122b_test",
            "general_reasoning_agent",
        }:
            raise ValueError(f"Unsupported general reasoning source: {source!r}")
        return {**route_audio_opd_row(row), "is_general_reasoning": False}
    if row.get("teacher_route") not in (None, "swe"):
        raise ValueError(f"General reasoning must use the SWE teacher: {source!r}")
    agent = row.get("agent_name")
    if agent not in (None, "", "general_reasoning_122b_test", "general_reasoning_agent", "single_turn_agent"):
        raise ValueError(f"Unexpected general reasoning agent {agent!r} for {source!r}")
    return {
        "teacher_route": "swe",
        "opd_source": "reasoning_" + source.removeprefix("122b_").removeprefix("benchmark_"),
        "agent_name": "general_reasoning_agent",
        "metric_data_source": row.get("metric_data_source") or source,
        "domain": "general",
        "is_general_reasoning": True,
    }


class ReasoningOPDMixedDataset(AudioOPDMixedDataset):
    """Add general routing before the SWE-only assumptions in the audio profile.

    Audio loading remains inherited verbatim. Prompts, images, references, and
    task identities are read from the original files without conversion.
    """

    def _read_files_and_tokenize(self):
        RLHFDataset._read_files_and_tokenize(self)
        self.dataframe = self.dataframe.map(route_reasoning_opd_row, desc="Assign SWE/audio/general OPD routes")
