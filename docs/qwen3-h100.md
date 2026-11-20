# Qwen3-VL-8B in one Slurm job (issue #3)

**The full evaluation has run, as run `job-37620`; see [Full run](#full-run-2026-11-19-to-20)
and the results in [docs/methods.md](methods.md#qwen3-vl-8b).** The history below led to it. Job 37600
stopped on seven of forty mismatching answers, before smoke or evaluation. Diagnostic 37609
restored 40/40 equality, including every saved logit, by excluding only cuDNN SDPA. This policy
is now integrated. Job **37614** passed the real, uninstrumented forty-question gate and stopped
in `--preflight-only` mode after 2min17s. No further execution is automatically authorized.
That job used the earlier launcher opt-in. The policy has since been centralized in the
subject constructor and passed another real 40/40 preflight in job **37619**. With full
execution authorized, 37619 advanced to smoke but failed in REVIS vector fitting on a
text-only image-placeholder validation error (4min53s, exit 1). It did not reach full evaluation.
The empty-image prompt is now tokenized directly without changing the template; evidence and
the compatibility departure are recorded in [docs/methods.md](methods.md).

The entry point is [experiments/qwen3_h100.sbatch](../experiments/qwen3_h100.sbatch). Submit it
from the repository root only after reviewing this checklist. It requests one H100 in `h100n2`,
one task, eight CPUs, 96 GiB of host memory and 48 hours. These are initial resource requests,
not measured requirements or a completion-time guarantee. Confirm the partition's time/memory
limits, account/QOS requirements, storage quota and network access before submission. The
script does not reserve a GPU until submitted and does not bypass Slurm.

**The launcher now defaults to `--preflight-only`.** This runs installation, checks and the
forty-question reference gate, then exits. Passing does not authorize smoke or full evaluation.
`--full` explicitly enables the remaining stages and requires separate authorization. For a
preflight-only submission, the Slurm time limit can be overridden to one hour. Arguments go
after the batch script; Slurm resource overrides go before it. Invalid arguments are refused.

## Qwen3 attention policy

The constructor of `HuggingFaceSubject` in [src/swce/models/hf.py](../src/swce/models/hf.py)
disables cuDNN SDPA for every wrapped model, without an environment variable or per-job
configuration. The native reference applies the same flag explicitly because it bypasses the
wrapper. The flag persists for fitting, calibration, generation, controls and diagnostics.

- Only `torch.backends.cuda.enable_cudnn_sdp(False)` is changed. This is not disabling cuDNN
  generally, not forcing eager attention, and not changing temperature, top-p, top-k, seed,
  bfloat16, token budgets or processor image budget.
- This is a **process-wide** flag, not a per-model property. Importing the module does nothing;
  constructing any `HuggingFaceSubject` disables it, including LLaVA and CPU/rehearsal subjects.
  Constructing a wrapper does not retroactively affect work done before construction. This
  broader default does not imply that GPU equivalence has been measured for all architectures.
- `HuggingFaceSubject.settings()` reads `torch.backends.cuda.cudnn_sdp_enabled()` into the
  top-level `cudnn_sdp_enabled` field, which evaluation records in every run's environment.
  It reads the effective flag, rather than a hardcoded value or a requested option. The
  preflight additionally records all backend flags. Neither record proves kernel dispatch.
- The initial environment, reference and SWCE attention flags must match for the gate to pass.
  Resume checks include the saved attention policy and launcher mode. Use a **new run ID**
  when changing code, policy or mode; do not resume 37600 or reuse its fitted artifacts.
- Validation is specific to the measured H100/software combination. Neither this flag nor
  seeding promises universal determinism. Keep the exact forty-question gate unchanged.

After centralizing the policy, shell syntax, Ruff lint/formatting, mypy (88 files) and the
offline suite passed: **180 passed, 10 skipped**, repeated inside job 37619. Obsolete opt-in
tests were replaced with constructor/forward and effective per-run metadata tests.

## Before authorizing execution

- No commit, staging or another person's approval is required to run locally. The job hashes
  tracked and new source/build files, including local edits, and saves a private source archive.
  Do not edit, pull or switch branches while it is queued or running: `sbatch` copies the shell
  script, not the Python source. The job guards the base commit, dependency lock and source
  manifest between stages, not every instant. This does not change the repository's PR rules.
- Make `uv` available in the batch shell as well as the interactive shell. The launcher searches
  `PATH`, the user's local bin directory and `SWCE_WORK_ROOT/tools/uv/bin`. On this machine,
  `uv 0.13.9` was installed in the latter isolated tool environment, without modifying system
  Python. The job stops if `uv` is missing; it does not download and execute an installer script.
- Confirm that the node's driver can run the repository's CUDA 13.0 PyTorch build. The job uses
  `uv sync --locked --python 3.13` in its own environment and never substitutes an older build,
  quantization, CPU inference, another accelerator or different generation parameters.
- Confirm that the compute node can download the locked dependencies, pinned checkpoint,
  benchmark files, recipe data and NLTK resources. Pinned downloads are checked by the existing
  data loader; COCO images have no published checksums. This does not run the registry-wide
  remote `verify`, which would also contact unrelated benchmarks.
- Reserve sufficient disk **and quota**, including inodes. The free-space check defaults to
  100 GiB as a conservative starting guard, not a measured dataset requirement; it cannot see
  all user quotas. The per-run environment, images, answers and diagnostic outputs persist.
- Complete environment and test validation before treating the preparation as ready for a
  full run. Another person's review is not a prerequisite of this local launcher.

## Files and isolation

By default everything is outside the repository, under `/raid/$USER/swce-qwen3-h100/`.
The similarly named directory under the user's home is on the home filesystem, not on RAID.

- `cache/`: uv, managed Python, Hugging Face and NLTK caches.
- `tools/uv/`: the isolated package-manager installation.
- `validation/venv/`: the environment used for CPU-only validation before submission.
- `runs/job-JOBID/source.tar.gz` and `source.sha256`: the exact source/build files and their
  hashes, including changes not committed. Deleted tracked files are marked in the manifest.
- `runs/job-JOBID/execution.txt`: authorized launcher mode. Attention policy is recorded in
  preflight/subject metadata, not configured by the shell job.
- `runs/job-JOBID/venv/`: the isolated locked environment; the project's existing environment
  is not modified.
- `runs/job-JOBID/data/`: benchmark data and `artifacts/`, including VTI directions and REVIS
  vectors/calibration with its per-block thresholds. Method and control share these artifacts.
- `runs/job-JOBID/preflight/`: environment, sampling settings, the forty native checkpoint
  answers, the forty SWCE answers and a comparison report. An interrupted validation must not
  be interpreted as passed.
- `runs/job-JOBID/results/smoke/`: smoke answers, scores and diagnostics, isolated by the
  existing experiment script.
- `runs/job-JOBID/results/`: the full `per_item`, leaderboard, analysis, shortcuts, lens and
  diagnostics, including probes. These are **not** the published results tables.
- `runs/job-JOBID/logs/JOBID/`: one log per stage, including driver and package checks.

`SWCE_WORK_ROOT` can select another absolute persistent storage root. Use storage outside git
or the ignored data directory, with adequate quota; the job must be able to create private
directories there. `SWCE_DATA`, environment and cache paths are deliberately set by the job,
not inherited from an unrelated experiment. Do not delete the run folder after scoring:
`per_item`, the recipe artifacts and diagnostics are required for analysis and reproducibility.
The initial Slurm log in the repository root is ignored separately. Do not log credentials.

## Sequential stages on the same allocated GPU

1. Record local source provenance, check the exclusive run-folder lock, disk and allocation.
2. Install the locked environment; run Ruff in check mode, mypy and offline pytest. Optional
   network rehearsals and official-repository tests are disabled, not represented as passed.
3. [experiments/qwen3_preflight.py](../experiments/qwen3_preflight.py) checks the actual GPU,
   CUDA/bfloat16 operation, software versions and sampling defaults. It loads the official
   pinned checkpoint natively with Transformers, then separately through `qwen3_vl_8b`, refusing
   missing/unexpected/mismatched weights and non-finite floating buffers in both paths. Exactly
   the first **40 questions** of POPE adversarial, not forty images, must match in decoded text
   and generated-token count, with identical prompts, images and generation configuration.
   This validates the SWCE adapter against the native checkpoint, not Transformers against
   an independently implemented model. Sampling mismatches block the run for investigation;
   do not switch to greedy decoding to make it pass.
  **The default `--preflight-only` mode stops here, even on success.**
4. Smoke on 16 images per task, with the real processor, methods and controls. The smoke runs
   the **full fitting/calibration recipes** when missing, so it may not be quick. It does not
   reduce the recipes or tune their parameters.
5. The full experiment: unmodified, VTI and control, REVIS and control, sequentially, using
   `code-default`, whole POPE adversarial, RePOPE adversarial, CHAIR and MME. Lens is recorded
   with every answer; residual shift uses fifty images. VTI is textual-only on Qwen; its visual
   branch is inapplicable. A recipe that is inapplicable remains a reportable outcome, not a
   reason to tune it or silently label the baseline as that method.
6. Explicit `--score-only`, with the default 1000 bootstrap resamples.
7. Linear probes of all POPE adversarial images, five folds over images, for baseline and the
   four method/control conditions the model can take.

Errors stop the script; it never automatically publishes or retries with different settings.
Inapplicable methods are logged by the existing pipeline and must be reviewed even if the
process exits successfully. A passed smoke alone is not proof of checkpoint correctness.

## Cluster inspection (2026-11-19)

- The current shell is on `dgx-H100-02`, without a Slurm allocation. No GPU computation is
  authorized outside a scheduled job. Pre-submission tests explicitly hide CUDA devices.
- The node reports H100 80 GB devices and driver `580.159.03`. This is the CUDA 13 driver
  series; the real PyTorch/bfloat16 operation passed in 37600 and the diagnostics, and is
  checked again inside every new allocation.
- The partition is up and reports unlimited partition-level wall time and memory. Account
  `default` allows QOS `onejob`, which declares no wall-time or memory limit and at most two
  running jobs per user. The script explicitly requests that account and QOS. Eligibility and
  resource availability are still determined by Slurm when the job is submitted.
- Actual RAID storage reports approximately 6.6 TiB free. The user quota is 500 GiB soft /
  600 GiB hard, with approximately 83 GiB used before installation. Availability is a snapshot,
  not a reservation. The installed tools, validation environment and caches occupy about 6 GiB.
- The repository's current path is on the home filesystem despite containing `raid` in its
  name. Large files therefore go to the actual RAID root, not below that home path.

## Initial offline validation (2026-11-19, before backend integration)

- `uv sync --locked --python 3.13` succeeded in the RAID validation environment. Neither
  `pyproject.toml` nor `uv.lock` was changed. No checkpoint or benchmark was downloaded.
- Installed versions: Python 3.13.18, PyTorch 2.14.1+cu130, torchvision 0.29.2, transformers
  5.23.0, Ruff 0.17.5, mypy 2.4.1 and pytest 9.2.0; package-manager version uv 0.13.9.
- Shell syntax, Ruff lint/formatting and mypy passed (85 source files checked by mypy).
- Pytest: **159 passed, 10 skipped**, in 13.13 seconds. Nine skips require clones of the
  official method repositories; one requires the network rehearsal. These checks remain
  unvalidated, not counted as passes.
- CUDA devices were hidden, CPU libraries limited to one thread and Hugging Face offline
  mode enabled for validation. No real-model loading, smoke, full run, probes or submission
  was performed. No commit or staging was required or performed.
- System Python is externally managed and lacks `ensurepip`. The existing pip installed uv
  into the dedicated tool environment using its `--python` option, without `sudo` or
  `--break-system-packages`. The batch launcher discovers that tool environment automatically.

## Diagnostic evidence (2026-11-19)

With all SDPA backends enabled, diagnostic 37607 matched 30/40 sampled reference/SWCE answers,
27/40 greedy answers, 31/40 same-model reference repeats and 32/40 fresh-load repeats. Inputs
and initial RNG states matched, but raw logits varied. A profile in replay 37608 observed
cuDNN SDPA on both paths for its first question; it does not cover every generation.

Job 37609 repeated the six diagnostic passes with only cuDNN SDPA disabled: **40/40 in all
four comparisons**, including all finite raw logits at every generated step (max difference
zero). It completed in 5min36s. Original sampling settings and the loading-context difference
were preserved. This supports an environment-specific workaround, not a universal root-cause
claim. No smoke, method fitting, full evaluation, scoring or probes have run on Qwen here.
See [docs/methods.md](methods.md) for the environment and protocol departure.

## Integrated preflight result (2026-11-19)

These measurements precede the constructor-only refactoring requested afterwards. They are
not a real-model validation of the new source revision; use a fresh run ID for that check.

- Job **37614** ran this launcher with explicit `--preflight-only`, a one-hour Slurm limit
  and a fresh run ID. It completed in **2min17s**, exit **0:0**, on one H100.
- Locked installation, shell syntax, Ruff lint/formatting, mypy (88 source files) and pytest
  passed. Both local and in-job suites reported **186 passed, 10 skipped**; the skips are the
  same nine optional official-repository tests and one network rehearsal, not new passes.
- The native checkpoint and SWCE produced **40/40 exactly equal answer records**, including
  IDs, images, prompts, full decoded text and generated-token counts. Generation configurations
  were identical, and the task fingerprint remained `59f90d40f5f9383a`.
- Initial, post-reference and post-SWCE attention flags matched: cuDNN SDPA off, other SDPA
  backends and general cuDNN on, deterministic algorithms off. Temperature 0.7, top-p 0.8,
  top-k 20, sampling, bfloat16 and seed zero stayed unchanged.
- This run used ordinary generation paths: no diagnostic monkeypatching, forced prefixes or
  raw-logit comparison. Full-logit equality was measured separately in diagnostic 37609;
  the production gate continues to require exact answer-record equality, not a relaxed test.
- The fresh private run folder contains the source archive/manifest, execution mode,
  environment, answers and stage logs. Only stages through `reference` ran; there is no results
  directory or fitted-artifact directory. No previous run was overwritten or resumed, no
  leaderboard rows were published, and the dependency lock did not change.

## Full run (2026-11-19 to 20)

- Job **37620** started `--full` at 03:44 on 2026-11-19 under run `job-37620`. Preflight (40/40)
  and smoke passed, and the unmodified condition finished: POPE and RePOPE in 61 minutes, CHAIR
  in 69 and MME in 3.5. At 07:08:48 the node ended every running job on it, those of other users
  included, with SIGTERM, while VTI was at question 568 of POPE.
- Job **37633** resumed it with `SWCE_RUN_ID=job-37620` at 13:04, after the base commit and the
  source manifest were checked to be identical. The runner continued VTI from question 568.
  Every condition, scoring and probes finished at 00:50 on 2026-11-20, exit 0:0, after 11h46.
- Generation runs at 40 to 50 tokens per second with one question at a time. A condition takes
  about 2h15, mostly POPE, whose answers reach the 64-token budget, and CHAIR. Conditions whose
  descriptions run into loops take longer on CHAIR, up to 14.5 seconds a description.
- Results: `/raid/$USER/swce-qwen3-h100/runs/job-37620/results/`. Its leaderboard rows are merged
  into `results/leaderboard.csv`.

## Interrupted runs and remaining scientific work

An interrupted run keeps its partial answers. To resume, explicitly reuse the previous
`SWCE_RUN_ID` with the same work root, base commit, source-file hashes, lock, launcher mode and
hardware/software/attention environment. The job locks that run identifier, refuses changed
provenance and reruns the
validation stages. The existing main script resumes completed items; probes and reference
checks are recomputed. Fitting/calibration may repeat if interrupted before saving an artifact.
Use a new identifier when code or environment changes; do not copy arbitrary old artifacts in.

This job prepares and runs the existing experiment and probes; it does **not** close issue #3
automatically. Before publishing, Henrick and Andre still need to record:

- REVIS selected block/threshold, method and control gate-opening rates at the first answer
  token on POPE and MME, and visual-vector signs across layers.
- CHAIR descriptions reaching the token budget and ending in loops, per condition; CHAIR gains
  against the matched-recall shortcut, marking incomparable controls.
- POPE and RePOPE F1 alongside yes ratio, AUROC and d-prime; MME cost and comparison of the
  unmodified model with the literature under the actual protocol.
- GPU, driver, Python, torch, torchvision, transformers, checkpoint revision, all departures
  and every inapplicable method/branch in [docs/methods.md](methods.md).

For run `job-37620`, [docs/methods.md](methods.md#qwen3-vl-8b) records all of these but one:
the block (34), threshold (-0.111) and signs of REVIS; the gate rates, inferred from the
first-token probabilities, as 100% for method and control on POPE and MME; the descriptions at
the budget and in loops, and the matched-recall shortcut, per condition; F1 with yes ratio,
AUROC and d'; the MME cost; and the environment. The comparison of the unmodified MME with the
literature is still open, since the Qwen3-VL technical report has not been checked.

Gate-rate and caption-loop analyses are not automated by this wrapper. Keep the captured
artifacts and outputs for those analyses; some diagnostics may require additional inference.
After review, merge **only full** rows into the published leaderboard and regenerate its
summary as [CONTRIBUTING.md](../CONTRIBUTING.md) describes. Never commit the private run folder.
