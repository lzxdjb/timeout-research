"""Instance-local legacy mbridge mappings for the opt-in audio projector."""


def configure_audio_projector_bridge(bridge):
    """Use the ordinary replicated-parameter load/export/checkpoint paths.

    Copy the mapping before extending it: mbridge implementations also expose
    this dictionary on the class, shared by unrelated actors and references.
    """
    if not isinstance(getattr(bridge, "_DIRECT_MAPPING", None), dict):
        raise ValueError("Audio projector training requires legacy mbridge direct parameter mappings")
    bridge._DIRECT_MAPPING = dict(bridge._DIRECT_MAPPING)
    for layer in ("proj1", "proj2"):
        for parameter in ("weight", "bias"):
            name = f"{layer}.{parameter}"
            bridge._DIRECT_MAPPING[f"_verl_audio_projector.{name}"] = f"model.audio_tower.{name}"
