rm -rf /cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/output/checkpoints/*

http://10.248.101.139:18080
http://10.248.101.131:18080
http://10.248.101.128:18080
http://10.248.100.187:18080
http://10.248.100.206:18080
http://10.248.100.182:18080
http://10.248.102.8:18080

############## grpo
export SWE_AGENT_TASK_FILTER_TRAINING_ENABLED=1
export SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD=0.25
export SWE_AGENT_TASK_FILTER_MIN_GROUP_ATTEMPTS=8
export SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES=1 ##### should set like that !!!!! this will also filter the data which is failed due to the queue

export SWE_AGENT_TRAINING_VERIFICATION_REWARD_SHAPING=1
export SWE_AGENT_TRAINING_VERIFICATION_PENALTY=0.1
export SWE_AGENT_TRAINING_VERIFICATION_WINDOW_TURNS=2
export SWE_AGENT_TRAINING_PROTOCOL_REWARD_SHAPING=1
export SWE_AGENT_TRAINING_SUBMISSION_PENALTY=0.1

export TRAIN_FILES='["/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/train.parquet"]'
export VAL_FILES='["/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_bench_pro.parquet","/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_bench_verified.parquet"]'
export SWE_AGENT_EXECUTION_URLS="http://10.248.101.139:18080,http://10.248.101.131:18080,http://10.248.101.128:18080,http://10.248.100.187:18080,http://10.248.100.206:18080,http://10.248.100.182:18080,http://10.248.100.194:18080"
export SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS=3
export EXPERIMENT_NAME=grpo_35B
# export SWE_AGENT_GENERATION_DEBUG=1
# export SWE_AGENT_TOOL_DEBUG=1
# export RAY_DEDUP_LOGS_ALLOW_REGEX='SWE_AGENT_(GENERATION|TOOL)_DEBUG|SWE_EXECUTION_ROUTER_DEBUG'
# export SWE_AGENT_MAX_TURNS=60

# Fail closed before starting training if any configured execution service is
# unavailable. Keep this check outside the trainer so a partial service outage
# cannot silently turn a rollout batch into zero-reward trajectories.
export SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS="${SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS:-5}"
if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required for SWE execution service health checks" >&2
  exit 1
fi
IFS=',' read -r -a _swe_execution_urls <<< "$SWE_AGENT_EXECUTION_URLS"
for _swe_execution_url in "${_swe_execution_urls[@]}"; do
  _swe_execution_url="${_swe_execution_url%/}"
  if [[ -z "$_swe_execution_url" ]]; then
    echo "ERROR: SWE_AGENT_EXECUTION_URLS contains an empty endpoint" >&2
    exit 1
  fi
  _swe_health_body="$(curl --noproxy '*' --fail --silent --show-error \
    --max-time "$SWE_AGENT_HEALTH_CHECK_TIMEOUT_SECONDS" \
    "${_swe_execution_url}/health")" || {
    echo "ERROR: SWE execution service is unreachable: ${_swe_execution_url}" >&2
    exit 1
  }
  _swe_health_compact="$(printf '%s' "$_swe_health_body" | tr -d '[:space:]')"
  if [[ "$_swe_health_compact" != *'"ok":true'* ]]; then
    echo "ERROR: SWE execution service is not healthy: ${_swe_execution_url}" >&2
    printf 'Health response: %s\n' "$_swe_health_body" >&2
    exit 1
  fi
  echo "SWE execution service healthy: ${_swe_execution_url}"
done
unset _swe_execution_url _swe_health_body _swe_health_compact _swe_execution_urls

export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROLLOUT_DATA_DIR="/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/rollout_data/${EXPERIMENT_NAME}"
mkdir -p "$ROLLOUT_DATA_DIR"

export VALIDATION_DATA_DIR="/cpfs01/nlp/leizhengxing/swe/verl/validation_data/${EXPERIMENT_NAME}"
mkdir -p "$VALIDATION_DATA_DIR"


nohup bash scripts/run_swe_agent_qwen3_5_35b_a3b.sh \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  >"./swe_35Btrain_sync.log" 2>&1 &

################# end 

################ ppo start

cd /cpfs01/nlp/leizhengxing/swe/verl

export MODEL_PATH="/cpfs01/nlp/leizhengxing/stock-rl-reflect/data/Qwen3.5-35-A3B"

export TP=2
export PP=1
export CP=2
export EP=8
export ETP=1
export GEN_TP=2
export NDEVICES_PER_NODE=8
export ALL_OFFLOAD=True

export TRAIN_FILES='["/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/train.parquet"]'
export VAL_FILES='["/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_bench_pro.parquet","/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_bench_verified.parquet"]'

export SWE_AGENT_EXECUTION_URLS="http://10.248.100.94:18080,http://10.248.100.208:18080,http://10.248.100.205:18080,http://10.248.100.203:18080,http://10.248.100.193:18080,http://10.248.100.87:18080,http://10.248.100.14:18080"

export SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS=3
export SWE_AGENT_TASK_FILTER_TRAINING_ENABLED=1
export SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES=1

export PROJECT_NAME="35A3B-SWE-PPO"
export EXPERIMENT_NAME="ppo_35B_swe_cp2"

export ROLLOUT_DATA_DIR="/cpfs01/nlp/leizhengxing/swe/verl/rollout_data/${EXPERIMENT_NAME}"
mkdir -p "$ROLLOUT_DATA_DIR"

export VALIDATION_DATA_DIR="/cpfs01/nlp/leizhengxing/swe/verl/validation_data/${EXPERIMENT_NAME}"
mkdir -p "$VALIDATION_DATA_DIR"

nohup bash scripts/run_swe_agent_qwen3_5_35b_a3b_ppo.sh \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  >"$ROLLOUT_DATA_DIR/training.log" 2>&1 &

echo "PID=$! log=$ROLLOUT_DATA_DIR/training.log"


#### ppo end









export TRAIN_FILES='["/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/train.parquet"]'
export VAL_FILES='["/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_bench_pro.parquet","/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_bench_verified.parquet"]'
# export SWE_AGENT_EXECUTION_URLS="http://10.248.100.97:18080,http://10.248.100.142:18080"
export SWE_AGENT_EXECUTION_URLS="http://10.248.102.0:18080,http://10.248.100.142:18080,http://10.248.102.1:18080,http://10.248.102.19:18080"
export SWE_AGENT_ROLLOUT_CLAIM_EXECUTOR_WORKERS=3
export SWE_AGENT_GENERATION_DEBUG=1
export SWE_AGENT_TOOL_DEBUG=1
export RAY_DEDUP_LOGS_ALLOW_REGEX='SWE_AGENT_(GENERATION|TOOL)_DEBUG|SWE_EXECUTION_ROUTER_DEBUG'
nohup bash scripts/run_swe_agent_qwen3_5_35b_a3b_async.sh \
  >"./swe_35Btrain_async" 2>&1 &


  pkill -KILL -f 'vllm' || true
  pkill -KILL -f 'VLLM' || true
  pgrep -f 'multiprocessing\.spawn.*spawn_main' | xargs -r kill -KILL
  rm -rf /tmp/ray_idle_digital_onboarding_4b_watchdog.lock
  cd /cpfs01/nlp/leizhengxing/swe/stock-rl-reflect
bash run_script/stop_ray_idle_digital_onboarding_4b.sh

  
cd /cpfs01/nlp/leizhengxing/swe/stock-rl-reflect
nohup bash run_script/monitor_ray_and_start_digital_onboarding_4b.sh \
  > /tmp/ray_idle_digital_onboarding_4b_watchdog.log 2>&1 &

echo $! > /tmp/ray_idle_digital_onboarding_4b_watchdog.pid
disown

bash run_script/stop_ray_idle_digital_onboarding_4b.sh


########## configuration of service model


export SWE_AGENT_DOCKERHUB_USERNAME="lzxdjb"
export SWE_AGENT_DOCKERHUB_TOKEN="dckr_pat_OKnY0wohlC0VuiI0M9IPVwlPq0c"

export RAYON_NUM_THREADS=8
export UV_THREADPOOL_SIZE=8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1


export SWE_AGENT_SYNTHESIZED_IMAGE_BUILD_RETRIES=3
export SWE_AGENT_SYNTHESIZED_IMAGE_BUILD_RETRY_INITIAL_SECONDS=2
export SWE_AGENT_SYNTHESIZED_IMAGE_BUILD_RETRY_MAX_SECONDS=30

export SWE_AGENT_EXECUTION_DETACH_SESSION=1
export SWE_AGENT_EXECUTION_OPERATION_DEBUG=1

export SWE_AGENT_DOCKER_COMMAND_DEBUG=1
export SWE_AGENT_DOCKER_COMMAND_HEARTBEAT_SECONDS=30

export SWE_AGENT_CONTAINER_RESOURCE_DEBUG=1
export SWE_AGENT_CONTAINER_RESOURCE_DEBUG_TIMEOUT_SECONDS=5

export SWE_AGENT_EXECUTION_HEALTH_LOCK_TIMEOUT_SECONDS=1
export SWE_AGENT_DOCKER_STATUS_TIMEOUT_SECONDS=5

export SWE_AGENT_EXECUTION_WATCHDOG=1
export SWE_AGENT_EXECUTION_WATCHDOG_INTERVAL_SECONDS=15
export SWE_AGENT_EXECUTION_WATCHDOG_TIMEOUT_SECONDS=5
export SWE_AGENT_EXECUTION_WATCHDOG_LOG=./swe_execution_watchdog_8_cpus.log

# export SWE_AGENT_EXTERNAL_PROXY="http://hexin:hx300033@10.217.180.65:30100"
# export SWE_AGENT_EXTERNAL_NO_PROXY="localhost,127.0.0.1,10.244.0.0/16,.svc,.cluster.local"

export SWE_AGENT_EXECUTION_HOST=0.0.0.0
export SWE_AGENT_EXECUTION_PORT=18080
export SWE_AGENT_EXECUTION_LISTEN_BACKLOG=1024
export SWE_AGENT_EXECUTION_MAX_CONCURRENT_CLAIMS=12

export SWE_AGENT_EXECUTION_MAX_CONCURRENT_EXECUTIONS=16
export SWE_AGENT_EXECUTION_MAX_CONCURRENT_HEAVY_EXECUTIONS=12
export SWE_AGENT_EXECUTION_ASYNC_ADMISSION_QUEUE=1
# Conservative shared cap for the 8-CPU service. Use 8 on a separately
# configured 16-CPU service after confirming CPU and memory headroom.
export SWE_AGENT_EXECUTION_MAX_CONCURRENT_CPU_HEAVY_OPERATIONS=16 #### share 4 cpus
export SWE_AGENT_EXECUTION_RESERVED_HEAVY_TOOL_SLOTS=6 #### reserve
export SWE_AGENT_EXECUTION_RESERVED_REWARD_SLOTS=6 #### reserve
export SWE_AGENT_EXECUTION_HEAVY_TOOL_DISPATCH_WEIGHT=3 ###### balance: 3:1 heavy tool: reward
export SWE_AGENT_EXECUTION_MAX_PENDING_HEAVY_EXECUTIONS=256
export SWE_AGENT_EXECUTION_MAX_PENDING_REWARDS=512
export SWE_AGENT_EXECUTION_QUEUED_OPERATION_ORPHAN_TTL_SECONDS=300
export SWE_AGENT_EXECUTION_OPERATION_POLL_INTERVAL_SECONDS=1

export SWE_AGENT_EXECUTION_MAX_ACTIVE_TRAJECTORIES=128
export SWE_AGENT_EXECUTION_CAPACITY_RETRY_AFTER_SECONDS=1
export SWE_AGENT_PREPARE_PARALLELISM=8

export SWE_AGENT_EXECUTION_SESSION_TTL_SECONDS=7200
export SWE_AGENT_EXECUTION_CLEANUP_INTERVAL=30

export SWE_AGENT_WORKSPACE_ROOT=/swe_benchmark_workspaces
export SWE_AGENT_DOCKER_DATA_ROOT=/swe-docker-data
export SWE_AGENT_DOCKER_STORAGE_DRIVER=overlay2
export SWE_AGENT_EXECUTION_USE_DOCKER=1
export SWE_AGENT_DOCKER_DISABLE_GPU=1
export SWE_AGENT_DOCKER_NETWORK_OVERRIDE=host

export SWE_AGENT_WORKSPACE_SEED_CACHE=1
export SWE_AGENT_WORKSPACE_SEED_CACHE_ROOT=/swe_benchmark_workspaces/.swe_seed_cache
export SWE_AGENT_WORKSPACE_SEED_CLONE_MODE=auto
export SWE_AGENT_WORKSPACE_SEED_CACHE_MAX_ENTRIES=32
export SWE_AGENT_WORKSPACE_SEED_CACHE_MAX_GB=0

export SWE_AGENT_EXECUTION_SEPARATE_REWARD_CAPACITY=1
export SWE_AGENT_EXECUTION_MAX_CONCURRENT_REWARDS=8
# export SWE_AGENT_EXECUTION_MAX_REWARD_SESSIONS=50

export SWE_AGENT_EXECUTION_ASYNC_WORKSPACE_CLEANUP=1
export SWE_AGENT_EXECUTION_WORKSPACE_CLEANUP_WORKERS=2
export SWE_AGENT_EXECUTION_MAX_PENDING_WORKSPACE_CLEANUPS=8
export SWE_AGENT_EXECUTION_WORKSPACE_CLEANUP_RETRIES=3
export SWE_AGENT_EXECUTION_WORKSPACE_MIN_FREE_GB=100

export SWE_AGENT_DOCKER_PULL_RETRIES=3
export SWE_AGENT_DOCKER_PULL_RETRY_INITIAL_SECONDS=2
export SWE_AGENT_DOCKER_PULL_RETRY_MAX_SECONDS=30
export SWE_AGENT_DOCKER_PULL_RETRY_AFTER_SECONDS=10


export SWE_AGENT_DOCKER_COMMAND_SILENT_TIMEOUT_SECONDS=600
# Server safety cap; clients request 600s for training and 900s for validation.
export SWE_AGENT_EXECUTION_TOOL_HARD_TIMEOUT_SECONDS=900 #### effective timeout = min(task timeout, phase-requested timeout, service-side cap)
export SWE_AGENT_EXECUTION_REWARD_HARD_TIMEOUT_SECONDS=900

export SWE_AGENT_DOCKER_PULL_MAX_PARALLELISM=12
export SWE_AGENT_IMAGE_PREFETCH_MAX_PARALLELISM=12
export SWE_AGENT_IMAGE_PREFETCH_PAUSE_FOR_CLAIMS=0


nohup bash run_script/start_swe_execution_service_for_benchmarks.sh \
  > ./swe_benchmark_execution_service_docker_32_cpus_1.log 2>&1 &


############## filter the data
export DATA_DIR=/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_filter
export SWE_AGENT_TASK_FILTER_MODE=report
export SWE_AGENT_TASK_FILTER_INFRA_RATIO_THRESHOLD=0.25
export SWE_AGENT_TASK_FILTER_MIN_ATTEMPTS=8
export SWE_AGENT_TASK_FILTER_INCLUDE_QUEUE_FAILURES=0
export SWE_AGENT_TASK_FILTER_ROLLOUT_N=8
export SWE_AGENT_TASK_FILTER_OUTPUT_DIR="$DATA_DIR/task_filter_reports_1"
export TRAIN_PARQUET=/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/data/swe_benchmarks/verl/swe_rebench_v2.parquet
export EXPERIMENT_NAME=grpo_35B
export ROLLOUT_DATA_DIR="/cpfs01/nlp/leizhengxing/swe/stock-rl-reflect/rollout_data/${EXPERIMENT_NAME}"

cd /cpfs01/nlp/leizhengxing/swe/stock-rl-reflect
python recipe/swe_agent/scripts/filter_training_tasks.py \
  --source-parquet "$TRAIN_PARQUET" \
  --trajectory-records "$ROLLOUT_DATA_DIR" \
  --output-dir "$SWE_AGENT_TASK_FILTER_OUTPUT_DIR" \
  --rollout-n "$SWE_AGENT_TASK_FILTER_ROLLOUT_N" \
  --mode apply
