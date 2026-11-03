"""Opt-in frozen audio conditioning before a Qwen3.5 bridge packs embeddings.

The encoder is deliberately not registered on the Megatron model: its immutable
weights remain in the source HF checkpoint and existing audio preservation/sync
continues to own them. Both supported bridges form full SBH embeddings before
their existing CP packing and TP sequence scatter. A scoped embedding hook adds
audio at that boundary without changing either bridge's distributed layout.
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
# Two one-time diagnostics are the Stage-1 evidence that audio is actually used.
# Keep them visible without increasing verbosity for any other verl module.
logger.setLevel(logging.INFO)
_ADAPTER_ATTRIBUTE = "_verl_frozen_audio_adapter"
_ENABLED_ATTRIBUTE = "_verl_frozen_audio_enabled"
_PREFIX = "model.audio_tower."
# The projector alone is registered before Megatron's dtype/DDP/optimizer setup.
# Its stable internal names also identify it in distributed checkpoints.
AUDIO_PROJECTOR_ATTRIBUTE = "_verl_audio_projector"
_PROJECTOR_NAMES = frozenset({"proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias"})


def _base_model(model):
    # DDP / Float16Module wrappers expose .module. Avoid importing the Megatron
    # stack so this boundary and its gradient semantics can be tested on CPU.
    while hasattr(model, "module"):
        model = model.module
    return model


def _embedding_boundary(model):
    language_model = getattr(model, "language_model", None)
    # Legacy mbridge shadows .embedding with a plain MTP scatter callable.
    # Its VL forward explicitly calls the registered, unscattered module.
    # Native Megatron-Bridge uses the same registered module through .embedding.
    registered = getattr(language_model, "_modules", {}).get("embedding")
    if isinstance(registered, torch.nn.Module):
        return registered
    embedding = getattr(language_model, "embedding", None)
    if not isinstance(embedding, torch.nn.Module):
        raise ValueError("frozen_audio requires the Qwen3.5 vision/language bridge embedding boundary")
    return embedding


def _audio_lengths(lengths):
    """Same 100-frame chunking formula as the Omni processor and audio tower."""
    remainder = lengths % 100
    return (((remainder - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1 + (lengths // 100) * 13


class FrozenAudioAdapter:
    """Lazy, fully frozen HF audio tower outside trainable/exported model state."""

    def __init__(self, source, audio_config, audio_token_id, hidden_size):
        self.source = Path(source)
        self.audio_config = dict(audio_config)
        self.audio_token_id = int(audio_token_id)
        self.hidden_size = int(hidden_size)
        if self.audio_config.get("output_dim") != self.hidden_size:
            raise ValueError("Frozen audio output dimension must match the student language hidden size")
        index = self.source / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            self.files = {
                name[len(_PREFIX) :]: self.source / file
                for name, file in weight_map.items()
                if name.startswith(_PREFIX)
            }
        elif (self.source / "model.safetensors").is_file():
            from safetensors import safe_open

            file = self.source / "model.safetensors"
            with safe_open(file, framework="pt", device="cpu") as handle:
                self.files = {name[len(_PREFIX) :]: file for name in handle.keys() if name.startswith(_PREFIX)}
        else:
            raise ValueError("frozen_audio requires a local HF safetensors source checkpoint")
        if not self.files or any(not file.is_file() for file in self.files.values()):
            raise ValueError("Frozen audio source checkpoint has missing audio tensors or shards")
        self.tower = None
        self._logged_forward = False

    def _load(self, device):
        from safetensors import safe_open
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

        config = Qwen3OmniMoeAudioEncoderConfig(**self.audio_config)
        # The native varlen FlashAttention path matches rollout's chunk boundaries.
        # CPU uses eager attention only for small numerical regression fixtures.
        config._attn_implementation = "flash_attention_2" if device.type == "cuda" else "eager"
        with torch.device("meta"):
            tower = Qwen3OmniMoeAudioEncoder(config)
        expected = set(tower.state_dict())
        if expected != set(self.files):
            raise ValueError(
                f"Frozen audio checkpoint schema mismatch: missing={sorted(expected - self.files.keys())}, "
                f"unexpected={sorted(self.files.keys() - expected)}"
            )
        state = {}
        for file in sorted(set(self.files.values())):
            with safe_open(file, framework="pt", device="cpu") as handle:
                for name, source_file in self.files.items():
                    if source_file == file:
                        state[name] = handle.get_tensor(_PREFIX + name)
        tower.load_state_dict(state, strict=True, assign=True)
        # This deterministic buffer is nonpersistent and therefore absent from
        # safetensors; materialize it explicitly after meta construction.
        positions = tower.positional_embedding
        tower.positional_embedding = type(positions)(positions.length, positions.channels, positions.max_timescale)
        tower.requires_grad_(False).eval()
        self.tower = tower.to(device)
        if hasattr(self, "projector"):
            logger.info(
                "Loaded frozen actor audio encoder for trainable projector: source=%s tensors=%d "
                "encoder_grad=False projector_grad=True",
                self.source,
                len(state),
            )
        else:
            logger.info(
                "Loaded frozen actor audio encoder and projector: source=%s tensors=%d parameters=%d "
                "encoder_grad=False projector_grad=False",
                self.source,
                len(state),
                sum(parameter.numel() for parameter in tower.parameters()),
            )

    def offload(self):
        if self.tower is not None:
            self.tower.to("cpu")

    def encode(self, features, mask, device):
        if (
            features.ndim != 3
            or mask.ndim != 2
            or features.shape[0] != mask.shape[0]
            or features.shape[-1] != mask.shape[-1]
        ):
            raise ValueError("Frozen audio requires aligned [audio, mel, frames] features and [audio, frames] masks")
        if features.shape[1] != self.audio_config["num_mel_bins"]:
            raise ValueError("Frozen audio mel dimensions do not match the source encoder")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Audio feature attention mask must contain only zero and one")
        lengths = mask.long().sum(-1)
        expected_mask = torch.arange(mask.shape[-1], device=mask.device)[None] < lengths[:, None]
        if not torch.equal(mask.bool(), expected_mask) or torch.any(lengths <= 0):
            raise ValueError("Audio feature attention mask must describe nonempty right-padded clips")
        if self.tower is None:
            self._load(device)
        self.tower.to(device).eval()
        with torch.no_grad():
            features = features.to(device=device, dtype=next(self.tower.parameters()).dtype)
            mask = mask.to(device=device, dtype=torch.bool)
            # HF's tower expects concatenated valid frames [mel, total_frames].
            packed = features.transpose(1, 2)[mask].transpose(0, 1).contiguous()
            output = self.tower(packed, feature_lens=lengths.to(device)).last_hidden_state
        expected = int(_audio_lengths(lengths).sum())
        if output.shape != (expected, self.hidden_size) or not torch.isfinite(output).all():
            raise ValueError("Frozen audio encoder returned invalid feature count, dimension, or nonfinite values")
        if not self._logged_forward:
            logger.info(
                "Frozen actor audio conditioning: clips=%d features_shape=%s mask_shape=%s embeddings=%d hidden=%d "
                "encoder_grad=False projector_grad=False",
                len(lengths),
                tuple(features.shape),
                tuple(mask.shape),
                expected,
                self.hidden_size,
            )
            self._logged_forward = True
        return output.detach()


class AudioProjector(torch.nn.Module):
    """The two original HF projection layers, with their original activation."""

    def __init__(self, audio_config, state):
        super().__init__()
        from transformers.activations import ACT2FN

        with torch.device("meta"):
            self.proj1 = torch.nn.Linear(audio_config["d_model"], audio_config["d_model"])
            self.proj2 = torch.nn.Linear(audio_config["d_model"], audio_config["output_dim"])
        self.act = ACT2FN[audio_config.get("activation_function", "gelu")]
        self.load_state_dict(state, strict=True, assign=True)
        for parameter in self.parameters():
            # The bridge scatters full embeddings across TP with an all-gather
            # backward. Every TP replica therefore already has the full gradient,
            # just like the bridge's replicated vision module.
            parameter.average_gradients_across_tp_domain = True

    def forward(self, hidden_states):
        return self.proj2(self.act(self.proj1(hidden_states)))


class TrainableAudioProjectorAdapter(FrozenAudioAdapter):
    """Frozen lazy encoder and a separately registered trainable HF projector.

    Only ``projector`` belongs to the Megatron model. The encoder, including
    ``ln_post``, stays outside its optimizer and distributed checkpoint state.
    """

    def __init__(self, source, audio_config, audio_token_id, hidden_size):
        super().__init__(source, audio_config, audio_token_id, hidden_size)
        from safetensors import safe_open

        if not _PROJECTOR_NAMES.issubset(self.files):
            raise ValueError("Trainable audio projector requires proj1/proj2 weights and biases")
        state = {}
        for file in sorted({self.files[name] for name in _PROJECTOR_NAMES}):
            with safe_open(file, framework="pt", device="cpu") as handle:
                for name in sorted(_PROJECTOR_NAMES):
                    if self.files[name] == file:
                        state[name] = handle.get_tensor(_PREFIX + name)
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError("Trainable audio projector checkpoint contains nonfinite weights")
        self.projector = AudioProjector(self.audio_config, state)

    def _load(self, device):
        super()._load(device)
        # Reuse the original HF encoder forward through ln_post. Removing only
        # these last operations avoids maintaining a second copy of HF chunking,
        # convolution, attention, or normalization logic.
        self.tower.proj1 = torch.nn.Identity()
        self.tower.act = torch.nn.Identity()
        self.tower.proj2 = torch.nn.Identity()

    def encode(self, features, mask, device):
        if (
            features.ndim != 3
            or mask.ndim != 2
            or features.shape[0] != mask.shape[0]
            or features.shape[-1] != mask.shape[-1]
        ):
            raise ValueError("Audio requires aligned [audio, mel, frames] features and [audio, frames] masks")
        if features.shape[1] != self.audio_config["num_mel_bins"]:
            raise ValueError("Audio mel dimensions do not match the source encoder")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Audio feature attention mask must contain only zero and one")
        lengths = mask.long().sum(-1)
        expected_mask = torch.arange(mask.shape[-1], device=mask.device)[None] < lengths[:, None]
        if not torch.equal(mask.bool(), expected_mask) or torch.any(lengths <= 0):
            raise ValueError("Audio feature attention mask must describe nonempty right-padded clips")
        if self.tower is None:
            self._load(device)
        self.tower.to(device).eval()
        projector_parameter = next(self.projector.parameters())
        if projector_parameter.device != device:
            raise ValueError("Registered audio projector must be on the actor device before forwarding")
        with torch.no_grad():
            features = features.to(device=device, dtype=next(self.tower.parameters()).dtype)
            mask = mask.to(device=device, dtype=torch.bool)
            packed = features.transpose(1, 2)[mask].transpose(0, 1).contiguous()
            encoded = self.tower(packed, feature_lens=lengths.to(device)).last_hidden_state
        # This is the only audio operation tracked by autograd. Casting bridges
        # an immutable source tower and Megatron's configured parameter dtype.
        output = self.projector(encoded.to(dtype=projector_parameter.dtype))
        expected = int(_audio_lengths(lengths).sum())
        if output.shape != (expected, self.hidden_size) or not torch.isfinite(output).all():
            raise ValueError("Audio projector returned invalid feature count, dimension, or nonfinite values")
        if not self._logged_forward:
            logger.info(
                "Trainable actor audio conditioning: clips=%d features_shape=%s mask_shape=%s "
                "embeddings=%d hidden=%d encoder_grad=False projector_grad=True projector_parameters=%d",
                len(lengths),
                tuple(features.shape),
                tuple(mask.shape),
                expected,
                self.hidden_size,
                sum(parameter.numel() for parameter in self.projector.parameters()),
            )
            self._logged_forward = True
        return output


def install_trainable_audio_projector_adapters(models, source, hf_config):
    """Register projectors before Megatron wraps models and builds optimizers.

    Call this from a model-creation callback, never on an already DDP-wrapped
    model. Later pipeline stages need only the enabled marker.
    """
    config = hf_config.audio_config
    config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    text_config = getattr(hf_config, "text_config", hf_config)
    adapters = []
    for model in models:
        if _base_model(model) is not model:
            raise ValueError("Trainable audio projector must be installed before model wrapping")
        setattr(model, _ENABLED_ATTRIBUTE, True)
        if not getattr(model, "pre_process", False):
            continue
        if hasattr(model, AUDIO_PROJECTOR_ATTRIBUTE) or hasattr(model, _ADAPTER_ATTRIBUTE):
            raise ValueError("Audio projector or adapter is already installed on this model")
        embedding = _embedding_boundary(model)
        if getattr(embedding, "scatter_to_sequence_parallel", False) or getattr(
            embedding, "reduce_scatter_embeddings", False
        ):
            raise ValueError("Audio must be inserted before embedding sequence-parallel scatter")
        adapter = TrainableAudioProjectorAdapter(source, config, hf_config.audio_token_id, text_config.hidden_size)
        setattr(model, AUDIO_PROJECTOR_ATTRIBUTE, adapter.projector)
        setattr(model, _ADAPTER_ATTRIBUTE, adapter)
        adapters.append(adapter)
        logger.info(
            "Registered trainable actor audio projector: source=%s parameters=%d "
            "encoder_grad=False projector_grad=True",
            source,
            sum(parameter.numel() for parameter in adapter.projector.parameters()),
        )
    return adapters


def install_frozen_audio_adapters(models, source, hf_config):
    """Attach plain adapter objects, preserving native parameter/export names."""
    config = hf_config.audio_config
    config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    text_config = getattr(hf_config, "text_config", hf_config)
    adapters = []
    for wrapped in models:
        model = _base_model(wrapped)
        setattr(model, _ENABLED_ATTRIBUTE, True)
        if not getattr(model, "pre_process", False):
            continue
        embedding = _embedding_boundary(model)
        if getattr(embedding, "scatter_to_sequence_parallel", False) or getattr(
            embedding, "reduce_scatter_embeddings", False
        ):
            raise ValueError("Frozen audio must be inserted before embedding sequence-parallel scatter")
        adapter = FrozenAudioAdapter(source, config, hf_config.audio_token_id, text_config.hidden_size)
        setattr(model, _ADAPTER_ATTRIBUTE, adapter)
        adapters.append(adapter)
    return adapters


@contextmanager
def frozen_audio_forward(model, input_ids, multi_modal_inputs):
    """Replace full-sequence audio embeddings, then let the bridge pack/scatter."""
    base = _base_model(model)
    if not getattr(base, _ENABLED_ATTRIBUTE, False):
        # Opt-in changes no forwarding behavior of existing text or audio jobs.
        yield
        return
    if not getattr(base, "pre_process", False):
        # Later PP stages receive already conditioned hidden states.
        yield
        return
    adapter = getattr(base, _ADAPTER_ATTRIBUTE, None)
    if adapter is None:
        raise ValueError("Enabled frozen audio preprocessing stage has no adapter")
    features = multi_modal_inputs.get("input_features")
    mask = multi_modal_inputs.get("feature_attention_mask")
    if features is None and mask is None:
        if torch.any(input_ids == adapter.audio_token_id):
            raise ValueError("Audio placeholders require audio features; refusing unconditioned actor scoring")
        yield
        return
    if features is None or mask is None:
        raise ValueError("Audio conditioning requires both input_features and feature_attention_mask")
    if input_ids.ndim != 2 or input_ids.is_nested:
        raise ValueError("Frozen audio requires full padded input IDs before bridge packing")
    audio_mask = input_ids == adapter.audio_token_id
    encoded = adapter.encode(features, mask, input_ids.device)
    if int(audio_mask.sum()) != len(encoded):
        raise ValueError("Audio placeholder count differs from frozen encoder features")
    # A global token count alone could silently attach one clip's frames to
    # another example. Match each contiguous placeholder run in row-major order.
    starts = (audio_mask & ~torch.nn.functional.pad(audio_mask[:, :-1], (1, 0))).nonzero()
    ends = (audio_mask & ~torch.nn.functional.pad(audio_mask[:, 1:], (0, 1))).nonzero()
    run_lengths = (ends[:, 1] - starts[:, 1] + 1).tolist()
    if run_lengths != _audio_lengths(mask.long().sum(-1)).tolist():
        raise ValueError("Audio placeholder runs do not match per-clip feature lengths and ordering")
    consumed = 0

    def replace(_module, args, kwargs, output):
        nonlocal consumed
        embedding_ids = kwargs.get("input_ids", args[0] if args else None)
        if consumed or not torch.equal(embedding_ids, input_ids):
            raise ValueError("Bridge must embed the complete audio input exactly once before packing")
        if not isinstance(output, torch.Tensor) or output.shape != (*reversed(input_ids.shape), adapter.hidden_size):
            raise ValueError("Bridge audio insertion requires complete [sequence, batch, hidden] embeddings")
        result = output.transpose(0, 1).clone()
        result[audio_mask] = encoded.to(dtype=output.dtype)
        consumed += 1
        return result.transpose(0, 1).contiguous()

    handle = _embedding_boundary(base).register_forward_hook(replace, with_kwargs=True)
    try:
        yield
        if consumed != 1:
            raise ValueError("Bridge did not consume frozen audio embeddings")
    finally:
        handle.remove()
