"""Migrate a trusted v2 OPD loader checkpoint using its original filtered Arrow snapshot.

Run with ``python -m scripts.audio_data.migrate_opd_sampler_checkpoint``. The output
is a separate ``global_step_N`` directory: model files and other checkpoint state
are symlinked, while only the loader's sampler identity/version are rewritten.
Resume with ``trainer.del_local_ckpt_after_load=false`` to preserve symlink targets.

The config must preserve the original quota mapping's insertion order. W&B may
sort these keys when writing YAML; restore their order from the original launch
command before migrating. A v2 identity did not encode quota key order, so it
cannot establish that the supplied order matches the original run.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import Dataset

from verl.utils.dataset.audio_opd_sampler import AudioOPDQuotaSampler


class _ArchivedPool:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_config(path):
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a data mapping")
    data = config.get("data", config)
    if isinstance(data, dict) and "opd_source_mix" not in data and isinstance(data.get("value"), dict):
        data = data["value"]
    if not isinstance(data, dict) or not {"opd_source_mix", "train_batch_size"} <= data.keys():
        raise ValueError("Config data must contain opd_source_mix and train_batch_size")
    if type(data["train_batch_size"]) is not int or data["train_batch_size"] <= 0:
        raise ValueError("Config train_batch_size must be a positive integer")
    return data


def _sampler_states(value, path=()):
    """Find the actual nested sampler state without assuming a torchdata release."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "sampler_state" and isinstance(child, dict):
                yield value, key, child, path + (key,)
            else:
                yield from _sampler_states(child, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _sampler_states(child, path + (index,))


def migrate_checkpoint(checkpoint, dataset, config, output):
    """Validate the original v2 identity, then create an independent resume directory."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    dataset = Path(dataset).resolve(strict=True)
    config = Path(config).resolve(strict=True)
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")
    output = output.resolve()
    if not checkpoint.is_dir() or not re.fullmatch(r"global_step_\d+", checkpoint.name):
        raise ValueError("Checkpoint must be a global_step_N directory")
    if output.name != checkpoint.name:
        raise ValueError("Output directory must have the same global_step_N name as the checkpoint")
    if output.is_relative_to(checkpoint):
        raise ValueError("Output must be outside the original checkpoint directory")
    if not (checkpoint / "actor").is_dir():
        raise ValueError("Original checkpoint has no actor directory")
    if os.path.lexists(checkpoint / "migration.json"):
        raise ValueError("Original checkpoint already contains migration.json")

    data = _data_config(config)
    dataframe = Dataset.from_file(str(dataset))
    sampler = AudioOPDQuotaSampler(_ArchivedPool(dataframe), data["opd_source_mix"], data["train_batch_size"])
    loader_path = checkpoint / "data.pt"
    state = torch.load(loader_path, map_location="cpu", weights_only=False)
    matches = list(_sampler_states(state))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one nested sampler_state, found {len(matches)}")
    parent, key, old_state, sampler_path = matches[0]
    if old_state.get("version") != 2:
        raise ValueError("Migration requires a version 2 sampler checkpoint")
    # Do not substitute a fingerprint supplied on the command line. The original
    # archive must itself reproduce the identity saved by the original sampler.
    sampler.load_state_dict(old_state)
    new_state = sampler.state_dict()
    if new_state.get("version") != 3:
        raise ValueError("Migration requires a sampler that saves version 3 state")
    for field, value in old_state.items():
        if field not in {"version", "identity"} and value != new_state.get(field):
            raise ValueError(f"Migration would change sampler field {field!r}")
    parent[key] = new_state

    provenance = {
        "migration_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "source_data_sha256": _file_sha256(loader_path),
        "dataset": str(dataset),
        "dataset_sha256": _file_sha256(dataset),
        "dataset_hf_fingerprint": dataframe._fingerprint,
        "dataset_content_fingerprint": sampler.dataset_fingerprint,
        "dataset_rows": len(dataframe),
        "config": str(config),
        "config_sha256": _file_sha256(config),
        "sampler_state_path": list(sampler_path),
        "old_version": old_state["version"],
        "old_identity": old_state["identity"],
        "new_version": new_state["version"],
        "new_identity": new_state["identity"],
        "schedule": sampler.schedule,
        "quota_source_order": [list(phase["quotas"]) for phase in sampler.schedule],
        "batch_size": sampler.batch_size,
        "output": str(output),
        "resume_del_local_ckpt_after_load": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        for child in sorted(checkpoint.iterdir()):
            if child.name != "data.pt":
                (output / child.name).symlink_to(child, target_is_directory=child.is_dir())
        torch.save(state, output / "data.pt")
        provenance["migrated_data_sha256"] = _file_sha256(output / "data.pt")
        with (output / "migration.json").open("x") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original global_step_N checkpoint")
    parser.add_argument("--dataset", type=Path, required=True, help="Original filtered Arrow snapshot")
    parser.add_argument(
        "--config", type=Path, required=True, help="Data config with original quota key order (restore sorted W&B keys)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Unused global_step_N directory outside original")
    args = parser.parse_args(argv)
    provenance = migrate_checkpoint(args.checkpoint, args.dataset, args.config, args.output)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
