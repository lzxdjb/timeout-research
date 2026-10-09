# SWE success-only repetition reward v1

The current judge prompt is `repeat-judge-v5`; reward composition and the frozen
`success-repeat-v1` fixture/label identities are unchanged.

The new policy is independent of `repeated_tool.py` and existing SFT filtering.
It is **off by default**. Do not enable the legacy repeated-tool reward flag at
same time. Partial-hidden-test and verification auxiliary reward modes are also
rejected when this mode is enabled, so reward composition cannot change silently.

| Context | Existing protocol reward | Repetition verdict | Applied reward |
|---|---:|---|---:|
| Training, valid raw task success (1) | 1 | repetitive | 0.1 |
| Training, valid raw task success (1) | 1 | clean / uncertain / judge error | 1 |
| Training, missing submission | -0.1 | bypass | -0.1 |
| Training, raw task failure (0 or -0.1) | any | bypass | unchanged |
| Validation or invalid evaluation | any | bypass | unchanged |
| Shadow mode | any | detection only | unchanged |

The multiplier defaults to 0.1 and is applied once to the positive existing
protocol reward. It is not a per-call deduction. Validation never calls the
judge or its preflight. In apply mode a model/context preflight runs before
training rollout execution; startup failure stops the run. Runtime judge errors
and uncertainty preserve the reward and record coverage/error metrics. They do
not create actor or infrastructure penalties.

## Components

- `stock-rl-reflect/recipe/swe_agent/repetition_reward.py`: independent policy,
  strict configuration/verdict validation, public prompt, async vLLM client,
  bounded state snapshots.
- `stock-rl-reflect/recipe/swe_agent/agent_loop.py`: dispatch evidence and reward
  composition after hidden evaluation. Releases execution session before judging.
- `stock-rl-reflect/recipe/swe_agent/remote_execution_service.py`: optional full
  observation hash and before/after read/search scope evidence.
- `stock-rl-reflect/recipe/swe_agent/tool_contract_info.py`: evidence switch and
  new source file included in service diagnostics.
- `verl/trainer/constants_ppo.py`: forwards new configuration and, when enabled,
  training reward settings to Ray workers for GRPO and PPO.
- `scripts/swe_repeat_regression.py` and `swe_repeat_contracts.py`: extend the
  existing benchmark with reviewed historical fixtures and judge contracts.
  New tests are in existing test files only.

## Level 1 and execution service deployment

The rule flags five consecutive identical completed repository reads/searches
only with identical normalized argument hashes, full observation hashes and
complete unchanged scope snapshots. Concurrent calls, mutations, transport
failures, unknown/truncated state, arbitrary shell/test/poll operations defer.
It never reports a trajectory as clean. Changing warning text does not invalidate
raw observation identity.

Both runtime dispatch and benchmark replay use the same public observation
classifier. A completed failed test or no-op edit is still eligible evidence;
transport failures are not. Unavailable search backends and incomplete output
artifacts are explicitly marked. Level 1 excludes these observations. Level 2
is instructed not to cite them, and a deterministic post-verdict guard withholds
a penalty if it does. Unrelated unavailable observations do not automatically
hide an independent, fully observed loop elsewhere in the same trace.

Existing services without this evidence remain usable: Level 1 defers and Level
2 judges eligible trajectories. To exercise Level 1 on live calls, restart the
execution services with the new code and this environment setting in their
startup environment:

```bash
export SWE_AGENT_REPEAT_EVIDENCE=1
```

This opt-in defaults to 0 to avoid silently adding filesystem work to existing
services. Snapshots include ignored/untracked content in the requested scope;
symlinks, external paths, artifacts, special files, time/size/file limits cause
deferral. Each snapshot is bounded to 1 second, 32 MiB, and 20,000 entries. Two
snapshots per read/search add overhead; monitor before broad rollout. Existing
source-consistency preflight checks require restarted services after deployment.

## Launch the judge (operator action; not performed during implementation)

Use a separate terminal on the H200 host with vLLM supporting Qwen3.5 MoE and
structured JSON output. The offline environment has vLLM 0.24.0 and Transformers
5.5.3. No package installation is performed by the launcher.

```bash
cd /cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl
export SWE_REPEAT_JUDGE_CHECKPOINT=/cpfs01/thscc/sharestorage/iwc/HithinkGPT/models/Qwen3.5-122
export SWE_REPEAT_JUDGE_GPUS=0,1,2,3
export SWE_REPEAT_JUDGE_TP=4
export SWE_REPEAT_JUDGE_HOST=0.0.0.0
export SWE_REPEAT_JUDGE_PORT=18090
# Set this to the judge pod IP or a routable Service DNS name.
export SWE_REPEAT_JUDGE_ADVERTISE_HOST=JUDGE_POD_IP_OR_SERVICE_DNS
export SWE_REPEAT_JUDGE_ADVERTISE_URL=http://JUDGE_POD_IP_OR_SERVICE_DNS:18090
export SWE_AGENT_REPEAT_JUDGE_MAX_CONTEXT=131072
export SWE_REPEAT_JUDGE_MAX_SEQS=4
export SWE_REPEAT_JUDGE_MEMORY_FRACTION=0.90
bash scripts/serve_swe_repeat_judge.sh --dry-run
bash scripts/serve_swe_repeat_judge.sh
```

The launcher binds vLLM on `0.0.0.0` and separately prints the client endpoint.
`0.0.0.0` is a listen address, never a client destination. The pod platform
must publish TCP port 18090 and allow traffic from the trainer pods: use Docker
host networking or `-p 18090:18090`, or a Kubernetes Service and NetworkPolicy
that route to the judge pod. A missing route, port publication, or firewall
rule cannot be repaired by the Python client or by changing the prompt.

Defaults:

- Checkpoint `/cpfs01/thscc/sharestorage/iwc/HithinkGPT/models/Qwen3.5-122`.
- BF16, no quantization, GPUs `0,1,2,3`, TP=4; the remaining four GPUs are unused.
- Served model `swe-repeat-judge`, port 18090, 131072 context, max 4 active sequences,
  GPU memory utilization 0.90, text-only, thinking disabled.
- Override with `SWE_REPEAT_JUDGE_CHECKPOINT`, `SWE_REPEAT_JUDGE_GPUS`,
  `SWE_REPEAT_JUDGE_TP`, `SWE_REPEAT_JUDGE_HOST`, `SWE_REPEAT_JUDGE_PORT`,
  `SWE_REPEAT_JUDGE_ADVERTISE_HOST`, `SWE_REPEAT_JUDGE_ADVERTISE_URL`,
  `SWE_REPEAT_JUDGE_MAX_SEQS`, `SWE_REPEAT_JUDGE_MEMORY_FRACTION` as needed.
  GPU count must equal TP.

Checkpoint configuration was inspected offline: Qwen3_5Moe architecture,
BF16, 262144 advertised context, no quantization configuration. Actual GPU
loading, long-context memory use and throughput still require online validation.

## Run the existing benchmark against the launched judge

On the trainer/diagnosis host (substitute the judge host IP):

```bash
cd /cpfs01/thscc/sharestorage/iwc/HithinkOmni/user_workspace/leizhengxing/leizhengxing/swe/verl
export PYTHONPATH="$PWD:$PWD/../stock-rl-reflect${PYTHONPATH:+:$PYTHONPATH}"
export SWE_AGENT_REPEAT_JUDGE_URL="http://JUDGE_HOST:18090"
python3 scripts/swe_repeat_regression.py judge-preflight
python3 scripts/swe_repeat_regression.py two-level-evaluate \
  --fixtures analysis/success_repeat_v1_benchmark/fixtures.jsonl.gz \
  --labels analysis/success_repeat_v1_benchmark/labels.json \
  --report analysis/success_repeat_v1_benchmark/online.json \
  --online
```

For a judge on the same host, use `http://127.0.0.1:18090`. For another pod,
set `SWE_AGENT_REPEAT_JUDGE_URL` to the printed pod IP or Service DNS endpoint,
never to `http://0.0.0.0:18090`.
The launcher occupies its terminal, so run the benchmark in a separate terminal.
To preserve a previous run, choose another report name, e.g. `online_v2.json`.

The corpus contains 19 real historical trajectories (including all 12 v9
trajectories) plus 6 contracts for polling, reread after edit, flaky verification,
transport recovery, prompt injection, and unavailable artifacts. Labels:
9 repetitive, 7 clean, 9 uncertain. These are evidence reviews by the code agent;
human adjudication is still needed before claiming calibrated accuracy. The
existing task-disjoint development/holdout split is preserved. There is no RM
training split because this implementation performs prompted inference only.

Reports include separate Level 1, residual Level 2, combined and raw-success
metrics; precision, recall, false-positive/negative rates, deferrals, uncertainty,
errors, latency and token usage. False-negative rate on explicit clean verdicts
and missed positives including deferrals are reported separately. Old raw-success
loops may fail today's submission gate: the raw-success subset is not an estimate
of how many current training trajectories will actually be penalized.

Online evaluation exits nonzero on a known-label mismatch, an error, or a
repetitive verdict on an uncertain-label control. Passing this small corpus is a
regression gate, not proof of universal accuracy: `production_ready` stays false.
Without `--online`, the command only replays Level 1 and makes no HTTP requests.
Historical traces lack trusted scope metadata, so all defer; this is not a
successful reward-model accuracy test.

The console summary now includes `regression_pass`, judge errors by failure
stage, unsafe uncertain-control flags, guard interventions and the historical
raw-success subset. `precision` and `false_positive_rate` use only clean/repetitive
labels; uncertain controls are reported separately. `false_negative_rate` counts
explicit wrong clean verdicts; `missed_positive_rate_including_deferral` also
counts withheld positive cases. Completion of the script is not a passing gate.

Prompt v5 requests just 2–4 representative event IDs, not all IDs in a long run.
The judge gives a concise evidence reason before choosing its verdict. Its
definitions explicitly separate an observed ineffective loop from missing
evidence; earlier useful work does not excuse a later established loop. It also
explicitly treats external-state polling that reaches completion as legitimate;
such polls need neither repository edits nor changing command arguments.
Short retries interleaved with test-path/configuration investigation and eventual
recovery are distinguished from long uninterrupted ineffective loops.
The JSON schema has separate repetitive/clean/uncertain branches, constraining
category and ID count consistently with the verdict. It limits evidence IDs to
four and requires a nonempty reason of at most 600 characters. Contradictory
replies are rejected; an uncertain verdict is never rewritten into repetitive.
Each request's schema restricts citations to actual completed, fully observed
event IDs. If fewer than two such events exist, the repetitive branch is removed.
This prevents the judge from citing an unexecuted terminal call or unavailable
observation; runtime validation and the evidence guard remain independent checks.

### Latest online check (2026-10-09)

`analysis/success_repeat_v1_benchmark/online_v5.json` records the current prompt's
25-case run: 9/9 repetitive labels detected, 7/7 clean labels preserved, zero
judge errors, and 5/5 repetitive historical raw-success cases detected. The
infrastructure-unavailable and missing-artifact controls are no longer penalized.
The overall gate **still fails**: `728813e61a077490a88f`, labelled uncertain,
is flagged for three invalid test commands interleaved with investigation and
eventual recovery. Keep apply disabled pending adjudication and a conservative
resolution of this recovery boundary. Do not relabel it merely to pass the gate.

Reports `online_v2.json` through `online_v5.json` preserve the diagnostic
iterations. They exposed contradictory verdict/category pairs, empty reasons,
an invalid unexecuted event citation, and sensitivity around polling/recovery.
These cases informed prompt/schema changes, so even the existing holdout is now
a regression set, not an untouched accuracy estimate. Use fresh task-disjoint
trajectories plus human-reviewed labels for independent calibration. No label
or frozen fixture content was changed during these iterations.
Malformed, incomplete or invalid replies preserve the reward and report the
failure stage, local validation detail, finish reason and completion-token usage,
including usage on failed completions. A bounded rejected model-response preview
(2048 characters), full response length and hash permit diagnosis without logging
the input transcript, HTTP credentials, or arbitrary network exception text.
Guarded proposals are retained for audit. Prompt/client changes require restarting
training workers, but do not require restarting the vLLM judge service. Restart
execution services before training when their source-consistency digest changes.

Rebuild a new immutable corpus version with:

```bash
python3 scripts/swe_repeat_regression.py two-level-build \
  --existing-fixtures analysis/diagnosis_20261007/repeat_regression_v1/fixtures.jsonl.gz \
  --v9-trajectories validation_data/val_only_tool_contract_v9_20261009T031709Z/0.jsonl \
  --output-dir analysis/success_repeat_v1_benchmark_NEW_VERSION
```

## Enable shadow, then apply after online review

Use the wrapper around your existing GRPO or PPO training command. Keep your
normal TRAIN_FILES, VAL_FILES, actor/critic and execution-service settings.

```bash
# Detection and audit only. No reward modifications.
bash scripts/run_swe_repeat_mode.sh shadow bash scripts/run_swe_agent_qwen3_5_35b_a3b.sh

# Only after reviewing the online benchmark and shadow results:
bash scripts/run_swe_repeat_mode.sh apply bash scripts/run_swe_agent_qwen3_5_35b_a3b.sh
```

The wrapper explicitly selects submission-only protocol shaping and a 0.1
missing-submission penalty. It refuses explicit incompatible reward settings;
set `SWE_AGENT_TRAINING_REPEATED_TOOL_REWARD_SHAPING=0` and disable the partial/aux
modes in the calling environment and in any existing script that overrides them.
For direct launches set `SWE_AGENT_REPEAT_REWARD_MODE=shadow|apply` after your
normal environment configuration. Set it to `off` to disable the new mode.

`SWE_AGENT_REPEAT_JUDGE_TIMEOUT` defaults to 120 seconds (includes queue wait),
`SWE_AGENT_REPEAT_JUDGE_CONCURRENCY` to 2 **per Ray worker/event loop**,
`SWE_AGENT_REPEAT_JUDGE_MAX_OUTPUT` to 1024 and
`SWE_AGENT_REPEAT_JUDGE_MAX_CONTEXT` to 131072. Server-side max-num-seqs bounds
active inference; many workers can still create queue pressure. Start small and
inspect latency, errors and judge coverage. Completion-ratio cutoffs can cancel
slow whole rollouts, so avoid aggressive cutoffs during initial shadow checks.

The server tokenizer counts the complete public prompt before inference. There
is no silent truncation: over-context traces are uncertain. The judge receives
only decoded model context plus an explicit allowlist of public event fields;
hidden tests, scores, gold patches and evaluator logs are never added. Artifact
pointers remain pointers; missing artifact contents may legitimately cause
uncertainty. The judge's output schema and completed event references are
validated; self-reported confidence is not treated as calibrated probability.

If authenticating the service, use vLLM's `VLLM_API_KEY` on the service host and
`SWE_AGENT_REPEAT_JUDGE_API_KEY_FILE` on the trainer pointing to a protected
shared file containing the key. Only its path is forwarded in Ray configuration.

Rollout reward metadata includes `repeat_reward_eligible`, `repeat_reward_applied`,
`repeat_reward_would_apply`, `repeat_reward_l1_flagged`, `repeat_reward_judge_called`,
`repeat_reward_uncertain`, errors, latency, tokens, original/final scores,
`repeat_reward_verdict_json`, and `repeat_public_events_json`. Judge audit records
contain prompt/policy versions, configured/served model identity and input hash.

## Offline checks

```bash
export PYTHONPATH="$PWD:$PWD/../stock-rl-reflect${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pytest -q \
  tests/scripts/test_swe_repeat_regression.py \
  tests/scripts/test_swe_tool_contract_benchmark.py \
  tests/scripts/test_swe_tool_online_benchmark.py \
  ../stock-rl-reflect/tests/utils/test_swe_agent_training_loop.py \
  ../stock-rl-reflect/tests/utils/test_swe_agent_remote_execution_service.py \
  ../stock-rl-reflect/tests/utils/test_swe_agent_tool_boundaries.py
```

These tests exercise reward truth tables, protocol composition, validation
bypass, release ordering, no-op and concurrent tool evidence, actual local
service read snapshots, mocked judge errors/overflow/schema/cancellation, public
projection and existing tool contracts. They do not load or assess Qwen weights.
