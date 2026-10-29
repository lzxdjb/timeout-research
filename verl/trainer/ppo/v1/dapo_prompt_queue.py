"""Original prompt carry-over for opt-in synchronous DAPO cutoff."""

import copy
import os
from collections import deque
from pathlib import Path
from uuid import uuid4

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


class DAPOPromptQueue:
    """Keep input templates separate from generated trajectories and policy versions."""

    def __init__(self):
        self.active: dict[str, TensorDict] = {}
        self.pending: deque[TensorDict] = deque()

    def __len__(self):
        return len(self.pending)

    def register(self, batch: TensorDict) -> None:
        for index, uid in enumerate(batch["uid"]):
            uid = str(uid)
            if uid in self.active:
                raise ValueError(f"Prompt already registered: {uid}")
            # Deep-copy non-tensor task/multimodal metadata too: workers may mutate it.
            self.active[uid] = copy.deepcopy(batch[index : index + 1]).cpu().clear_device_()

    def release(self, uids) -> None:
        for uid in uids:
            self.active.pop(uid, None)

    def requeue(self, uids) -> int:
        requested = set(uids)
        missing = requested - self.active.keys()
        if missing:
            raise RuntimeError(f"Missing original prompt templates for {len(missing)} DAPO groups")
        # Preserve submission order, independent of set ordering in ReplayBuffer.
        ordered = [uid for uid in self.active if uid in requested]
        self.pending.extend(self.active.pop(uid) for uid in ordered)
        return len(ordered)

    def take(self, count: int) -> list[TensorDict]:
        rows = [self.pending.popleft() for _ in range(min(count, len(self.pending)))]
        for row in rows:
            tu.assign_non_tensor_stack(row, "uid", [str(uuid4())])
        return rows

    def save(self, path: str) -> None:
        if self.active:
            raise RuntimeError("Cannot checkpoint DAPO prompt queue while rollout templates are active")
        destination = Path(path)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            torch.save({"version": 1, "pending": list(self.pending)}, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path: str) -> None:
        self.active.clear()
        self.pending.clear()
        if not os.path.exists(path):
            return  # Backward-compatible with checkpoints predating prompt carry-over.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("version") != 1:
            raise ValueError("Unsupported DAPO prompt queue checkpoint version")
        self.pending.extend(state["pending"])
