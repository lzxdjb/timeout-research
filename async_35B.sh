cd /cpfs01/nlp/leizhengxing/verl

WANDB_API_KEY=${WANDB_API_KEY:-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}
wandb login "$WANDB_API_KEY"


export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"


export PYTHONPATH=/cpfs01/nlp/leizhengxing/verl-async-deps:/cpfs01/nlp/leizhengxing/verl:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0


export MODEL_PATH=/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/Qwen3.5-35-A3B
export TRAIN_FILE='["/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/frontier/openmath_29.parquet","/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/frontier/math_mixture_42.parquet","/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/frontier/arxivqa_136.parquet","/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/frontier/deepvision_227.parquet","/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/knowledge-training/verl/medmcqa.parquet","/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/knowledge-training/verl/scibench.parquet","/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/knowledge-training/verl/super_gpqa.parquet"]'


export VAL_FILES="[/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/reasoning-benchmarks/verl/aime_2024.parquet,/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/reasoning-benchmarks/verl/aime_2025.parquet,/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/reasoning-benchmarks/verl/gpqa_diamond.parquet,/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/reasoning-benchmarks/verl/mmlu_pro.parquet,/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/reasoning-benchmarks/verl/bbh.parquet,/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/restored/reasoning-benchmarks/verl/hle.parquet]"


python3 -X faulthandler \
  -m verl.experimental.fully_async_policy.fully_async_main \
  --config-path=config \
  --config-name=fully_async_ppo_megatron_trainer.yaml \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size=0 \
  data.gen_batch_size=1 \
  data.val_batch_size=32 \
  data.max_prompt_length=8192 \
  data.max_response_length=40000 \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.optim.lr_decay_steps=3000 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48192  \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
  actor_rollout_ref.actor.megatron.use_remove_padding=True \
  actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.context_parallel_size=2 \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=4 \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
  actor_rollout_ref.actor.megatron.param_offload=True \
  actor_rollout_ref.actor.megatron.grad_offload=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.dtype=bfloat16 \
  actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48192  \
  actor_rollout_ref.ref.megatron.use_mbridge=True \
  actor_rollout_ref.ref.megatron.vanilla_mbridge=True \
  actor_rollout_ref.ref.megatron.use_remove_padding=True \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.ref.megatron.context_parallel_size=2 \
  actor_rollout_ref.ref.megatron.expert_model_parallel_size=4 \
  actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
  actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.max_num_seqs=2048 \
  actor_rollout_ref.rollout.max_model_len=48192  \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48192  \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.nccl_timeout=9600 \
  actor_rollout_ref.model.use_fused_kernels=True \
  reward.custom_reward_function.path=/cpfs01/nlp/leizhengxing/stock-rl-reflect/recipe/122B-test/reward_122b_test.py \
  reward.custom_reward_function.name=compute_score \
  trainer.logger="[console,wandb]" \
  trainer.project_name=35A3B-frontier-GRPO \
  trainer.experiment_name=35a3b_math_frontier_async_4train_4rollout \
  trainer.default_local_dir=/cpfs01/nlp/leizhengxing/stock-rl-reflect/output/checkpoints/35a3b_math_frontier_async_4train_4rollout \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=15 \
  trainer.val_before_train=False \
  trainer.test_freq=5 \
  trainer.save_freq=20 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=auto \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=4 \
  rollout.total_rollout_steps=3000 \
  async_training.use_dynamic_resource_scheduling=True \
  async_training.staleness_threshold=0.5 \
  async_training.trigger_parameter_sync_step=1 \
  async_training.require_batches=1 \
  async_training.partial_rollout=True \
  async_training.concurrent_samples_per_replica=16 \
  ray_kwargs.ray_init.num_cpus=128 \
  +reward.custom_reward_function.reward_kwargs.return_dict=True \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_inflight_gen_batches=1