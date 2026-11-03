"""Opt-in routing metadata for mixed SWE/audio OPD; source files remain unchanged."""

import asyncio

from verl.utils.dataset.rl_dataset import RLHFDataset


def route_audio_opd_row(row):
    """Add routing fields without changing prompts, references or SWE task identities."""
    source = row.get("data_source")
    if not isinstance(source, str) or not source:
        raise ValueError("Mixed OPD requires a nonempty data_source on every row")
    audio = source.startswith("audio_")
    route = "audio" if audio else "swe"
    if row.get("teacher_route") not in (None, route):
        raise ValueError(f"Conflicting teacher_route for {source!r}")
    agent = row.get("agent_name") or ("single_turn_agent" if audio else "swe_tool_agent")
    expected = "single_turn_agent" if audio else "swe_tool_agent"
    if agent != expected:
        raise ValueError(f"Mixed OPD expected {expected!r} for {source!r}, got {agent!r}")
    if audio and not row.get("audios"):
        raise ValueError(f"Audio OPD row {source!r} has no audio")
    return {
        "teacher_route": route,
        "opd_source": source if audio else "swe",
        "agent_name": agent,
        "metric_data_source": row.get("metric_data_source") or source,
        "domain": row.get("domain") or route,
    }


class AudioOPDMixedDataset(RLHFDataset):
    """Use only from the multi-teacher launcher; ordinary RLHFDataset is unchanged."""

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        self.dataframe = self.dataframe.map(route_audio_opd_row, desc="Assign SWE/audio OPD routes")

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        images, videos, audios = super()._process_multi_modal_info(messages, image_patch_size, config)
        if audios is not None:
            from verl.utils.tokenizer.audio_processor import load_audio_waveform

            sampling_rate = config.get("audio_sampling_rate", 16000)
            if type(sampling_rate) is not int or sampling_rate <= 0:
                raise ValueError("Audio OPD audio_sampling_rate must be a positive integer")
            # vLLM accepts waveforms with a rate, rather than paths. Use the same
            # samples in student feature extraction and generation. Original
            # prompt paths stay intact for teacher requests and diagnostics.
            audios = [
                (load_audio_waveform(audio, target_sampling_rate=sampling_rate), sampling_rate) for audio in audios
            ]
        return images, videos, audios

    @classmethod
    async def process_multi_modal_info(cls, messages, image_patch_size, config):
        return await asyncio.to_thread(cls._process_multi_modal_info, messages, image_patch_size, config)
