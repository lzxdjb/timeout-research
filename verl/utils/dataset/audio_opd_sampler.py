"""Stateful source quotas for the explicitly enabled synchronous audio OPD run."""

import copy
import hashlib
import json
import random
from collections import defaultdict

from torch.utils.data import Sampler


class AudioOPDQuotaSampler(Sampler):
    """Shuffle/cycle each source independently, keeping all state in the loader checkpoint.

    The V1 trainer supplies the optimizer-step number before each fetch. A complete
    quota round has ``batch_size`` prompt draws; infrastructure refills can consume
    additional rounds. Ratios describe requested prompts, not accepted trajectories.
    Zero loader workers are required so a worker cannot prefetch a future phase.
    """

    def __init__(self, dataset, schedule, batch_size, seed=0):
        from omegaconf import OmegaConf

        self.dataset = dataset
        self.batch_size = int(batch_size)
        if OmegaConf.is_config(schedule):
            schedule = OmegaConf.to_container(schedule, resolve=True)
        self.schedule = json.loads(schedule) if isinstance(schedule, str) else copy.deepcopy(list(schedule))
        if not self.schedule:
            raise ValueError("Audio OPD requires a nonempty source mix schedule")
        self.groups = defaultdict(list)
        self.index_sources = list(dataset.dataframe["opd_source"])
        for index, source in enumerate(self.index_sources):
            self.groups[source].append(index)
        for phase in self.schedule:
            if set(phase) != {"steps", "quotas"}:
                raise ValueError("Each OPD phase requires exactly steps and quotas")
            if type(phase["steps"]) is not int or phase["steps"] <= 0:
                raise ValueError("OPD phase steps must be positive integers")
            quotas = phase["quotas"]
            if not isinstance(quotas, dict) or not quotas:
                raise ValueError("OPD quotas must be a nonempty mapping")
            if any(type(value) is not int or value < 0 for value in quotas.values()):
                raise ValueError("OPD quotas must be nonnegative integers")
            if sum(quotas.values()) != self.batch_size:
                raise ValueError(f"OPD quotas must sum to train_batch_size={self.batch_size}")
            if any(count and not self.groups.get(source) for source, count in quotas.items()):
                raise ValueError("OPD schedule requests an empty or unknown source")
        identity = {
            "fingerprint": getattr(dataset.dataframe, "_fingerprint", None),
            "groups": dict(self.groups),
            "schedule": self.schedule,
            "batch_size": self.batch_size,
        }
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.rng = random.Random(0 if seed is None else seed)
        self.orders = {source: list(indices) for source, indices in self.groups.items()}
        self.cursors = {source: 0 for source in self.orders}
        for order in self.orders.values():
            self.rng.shuffle(order)
        self.step = 1
        self.phase_index = 0
        self.pending = []
        self.draw_counts = defaultdict(int)

    def set_step(self, step):
        step = int(step)
        if step < 1:
            raise ValueError("OPD training steps start at 1")
        end = 0
        for phase_index, phase in enumerate(self.schedule):
            end += phase["steps"]
            if step <= end:
                if phase_index != self.phase_index:
                    self.pending = []
                self.phase_index = phase_index
                self.step = step
                return
        raise ValueError(f"OPD step {step} exceeds the configured {end}-step curriculum")

    def _draw(self, source):
        order = self.orders[source]
        cursor = self.cursors[source]
        if cursor == len(order):
            self.rng.shuffle(order)
            cursor = 0
        index = order[cursor]
        self.cursors[source] = cursor + 1
        return index

    def __iter__(self):
        for _ in range(len(self)):
            if not self.pending:
                sources = [
                    source for source, count in self.schedule[self.phase_index]["quotas"].items() for _ in range(count)
                ]
                self.rng.shuffle(sources)
                # Reserve source slots, not dataset rows. A phase change may
                # discard unrequested slots; it must not advance source pools
                # past rows that have never actually been yielded.
                self.pending = sources
            source = self.pending.pop(0)
            index = self._draw(source)
            self.draw_counts[source] += 1
            yield index

    def __len__(self):
        return len(self.dataset)

    def state_dict(self):
        return copy.deepcopy(
            {
                "version": 2,
                "identity": self.identity,
                "rng": self.rng.getstate(),
                "orders": self.orders,
                "cursors": self.cursors,
                "step": self.step,
                "phase_index": self.phase_index,
                "pending": self.pending,
                "draw_counts": dict(self.draw_counts),
            }
        )

    def load_state_dict(self, state):
        if state.get("version") != 2 or state.get("identity") != self.identity:
            raise ValueError("Cannot resume OPD sampler with different data, batch size or curriculum")
        self.rng.setstate(state["rng"])
        self.orders = copy.deepcopy(state["orders"])
        self.cursors = copy.deepcopy(state["cursors"])
        self.step = state["step"]
        self.phase_index = state["phase_index"]
        self.pending = list(state["pending"])
        self.draw_counts = defaultdict(int, state["draw_counts"])
