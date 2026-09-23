# Frozen audio weights during RL rollout

The Megatron Qwen3.5 actor exports vision/text parameters. Its frozen Omni
audio tower is preserved in HF checkpoints separately. Enabling the audio
architecture alone does not load those parameters when the rollout uses
`load_format=dummy`, and vLLM level-2 sleep discards them again.

`vllm_rollout/frozen_audio.py` restores the tower after each complete IPC
weight update, including the initial actor synchronization, before weight
post-processing, cache invalidation, and generation resume. Level-2 sleep
and the actor's updated language/vision parameters remain unchanged.

## Configuration and checkpoint source

The existing configuration is sufficient:

```bash
actor_rollout_ref.rollout.enable_audio=True \
+trainer.audio_benchmark.enabled=True
```

Restoration applies to the `GageQwen3_5OmniMoeForConditionalGeneration`
architecture. It reads only `model.audio_tower.*` tensors from the local
model directory used by the vLLM worker (`model_config.model`). Both
`model.safetensors` and indexed, sharded safetensors checkpoints are supported.
There is no remote download or separate restoration flag.

The audio source is the configured model directory, including after a
training resume. It must contain the intended frozen tower. The current
step's language/vision weights come from the actor, not that source directory.
Complete audio parameters sent by an actor take precedence. A partial audio
update raises an error rather than mixing actor and checkpoint parameters.

Missing tensors, missing shards, incompatible shapes, and incomplete loader
coverage raise errors. Coverage validation checks Q, K, and V separately
before loading into vLLM's packed parameters. vLLM's native loader handles
tensor-parallel partitioning. Restoration streams CPU tensors and does not
keep a persistent second audio model in memory.

Each successful checkpoint restoration emits an INFO log:

```text
Restored frozen Omni audio: rank=0 step=4 source=... checkpoint_tensors=397 parameters=301 seconds=...
```

Counts depend on the checkpoint architecture. For the current 24-layer tower,
397 checkpoint tensors map to 301 runtime parameters because Q/K/V are packed.
Loading time includes checkpoint reads and tensor copies for that worker.

## Applying and verifying a change

Restart the training/rollout worker processes to load updated Python code.
An existing process does not acquire this fix by editing source files. A
restart can resume the existing saved training checkpoint through the normal
resume configuration.

CPU regressions:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/workers/rollout/test_frozen_audio_on_cpu.py \
  tests/workers/rollout/test_vllm_weight_update_utils_on_cpu.py \
  tests/utils/test_vllm_weight_name_normalization_on_cpu.py \
  tests/trainer/ppo/test_audio_benchmark_on_cpu.py

CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/workers/rollout/test_frozen_audio_native_loader_on_cpu.py
```

The native-loader test exercises real vLLM parameter classes and TP loaders
for both TP=2 partitions, replacing process-group metadata and unused forward
kernels so it runs on CPU. It checks exact values after initial restoration
and repeated simulated weight loss. It does not exercise CUDA memory sleep
or audio inference.

On available test GPUs, verify the initial restore and a complete
sleep/wake/actor-update cycle with known audio examples. Compare audio
parameters against the checkpoint partitions, verify real audio produces
appropriate transcription/reasoning responses, and include an altered-audio
control and text/vision regression. Use a small sample set before launching
the full benchmark suite. Restoring weights is necessary but does not by
itself validate preprocessing, embedding insertion, or end-to-end accuracy.
