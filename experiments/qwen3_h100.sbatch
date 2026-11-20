#!/usr/bin/env bash
# Submit from the repository root, only after reviewing docs/qwen3-h100.md.
# These resources are a starting request, not a validated runtime/memory estimate.
#SBATCH --job-name=swce-qwen3
#SBATCH --partition=h100n2
#SBATCH --account=default
#SBATCH --qos=onejob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=swce-qwen3-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
stage=initialization
trap 'code=$?; printf "FAILED: %s (exit %s). No publication was performed.\n" "$stage" "$code" >&2; exit "$code"' ERR

die() {
    printf 'STOP: %s\n' "$*" >&2
    exit 1
}

# Safe default: a successful reference check never authorizes the long experiment.
[[ "$#" -le 1 ]] || die 'Usage: qwen3_h100.sbatch [--preflight-only|--full]'
mode="${1:---preflight-only}"
case "$mode" in
    --preflight-only|--full) ;;
    *) die 'Usage: qwen3_h100.sbatch [--preflight-only|--full]' ;;
esac

# Refuse execution on a login node. Slurm, not this script, selects the GPU.
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]] || die 'Run only inside a Slurm allocation.'
cd "$SLURM_SUBMIT_DIR"
[[ -f experiments/main.py && -f uv.lock ]] || die 'Submit from the repository root.'
work_root="${SWCE_WORK_ROOT:-/raid/$USER/swce-qwen3-h100}"
[[ "$work_root" == /* ]] || die 'SWCE_WORK_ROOT must be absolute.'
# A user-local uv install need not be on a non-interactive shell's default PATH.
if ! command -v uv >/dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null && [[ -x "$work_root/tools/uv/bin/uv" ]]; then
    export PATH="$work_root/tools/uv/bin:$PATH"
fi
for command in uv git flock sha256sum df nvidia-smi sort tar; do
    command -v "$command" >/dev/null || die "Required command is missing: $command"
done
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die 'Slurm did not expose an allocated GPU.'
commit=$(git rev-parse HEAD)
lock_hash=$(sha256sum uv.lock)

# Include local edits and new source files without requiring a commit or staging.
# Limit the archive to code and build inputs; never copy arbitrary untracked secrets.
source_paths() {
    git ls-files --cached --others --exclude-standard -z -- \
        src methods experiments tests pyproject.toml uv.lock .python-version README.md LICENSE \
        | sort -zu
}

source_manifest() {
    source_paths | while IFS= read -r -d '' path; do
        [[ ! -L "$path" ]] || die "Symlink in source is not supported: $path"
        if [[ -f "$path" ]]; then
            sha256sum -- "$path" || return 1
        else
            printf 'MISSING %q\n' "$path"
        fi
    done
}

source_hash=$(source_manifest | sha256sum)

# A fresh run owns its data, fitted artifacts, environment and results. Reusing a run
# identifier is an explicit resume; caches alone are shared with other run identifiers.
run_id="${SWCE_RUN_ID:-job-$SLURM_JOB_ID}"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || die 'Invalid SWCE_RUN_ID.'
run_root="$work_root/runs/$run_id"
mkdir -p "$run_root"
exec 9>"$run_root/.lock"
flock -n 9 || die 'Another job owns this run identifier.'
provenance=$(printf '%s\n%s\n%s\n' "$commit" "$lock_hash" "$source_hash")
if [[ -f "$run_root/source.txt" ]]; then
    [[ "$(cat "$run_root/source.txt")" == "$provenance" ]] || die 'Code/lock changed: use a new SWCE_RUN_ID.'
else
    [[ -z "$(find "$run_root" -mindepth 1 -maxdepth 1 ! -name .lock -print -quit)" ]] || die 'Existing run has no provenance; use a new SWCE_RUN_ID.'
    printf '%s\n' "$provenance" >"$run_root/source.txt"
fi

if [[ ! -f "$run_root/source.tar.gz" ]]; then
    source_paths | while IFS= read -r -d '' path; do
        if [[ -f "$path" ]]; then
            printf '%s\0' "$path"
        fi
    done | tar --create --gzip --file="$run_root/source.tar.gz.part" \
        --null --verbatim-files-from --files-from=-
    mv "$run_root/source.tar.gz.part" "$run_root/source.tar.gz"
fi
source_manifest >"$run_root/source.sha256"

# Do not silently turn a validation-only run into a full run through resume.
execution=$(printf 'mode=%s' "$mode")
if [[ -f "$run_root/execution.txt" ]]; then
    [[ "$(cat "$run_root/execution.txt")" == "$execution" ]] || die 'Execution mode changed: use a new SWCE_RUN_ID.'
else
    printf '%s\n' "$execution" >"$run_root/execution.txt"
fi

minimum_gib="${SWCE_MIN_FREE_GIB:-100}"
[[ "$minimum_gib" =~ ^[1-9][0-9]*$ ]] || die 'SWCE_MIN_FREE_GIB must be positive.'
available_kib=$(df -Pk "$run_root" | awk 'END {print $4}')
(( available_kib >= minimum_gib * 1024 * 1024 )) || die 'Insufficient free disk space; also check your storage quota.'

export SWCE_DATA="$run_root/data"
export UV_PROJECT_ENVIRONMENT="$run_root/venv"
export UV_CACHE_DIR="$work_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$work_root/cache/python"
export HF_HOME="$work_root/cache/huggingface"
export NLTK_DATA="$work_root/cache/nltk"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Keep the test suite offline; its optional model rehearsals are not this preflight.
unset SWCE_NETWORK SWCE_OFFICIAL
logs="$run_root/logs/$SLURM_JOB_ID"
results="$run_root/results"
mkdir -p "$logs" "$SWCE_DATA" "$NLTK_DATA" "$run_root/preflight"

check_source() {
    [[ "$(git rev-parse HEAD)" == "$commit" ]] || die 'Checkout changed while the job was running.'
    [[ "$(sha256sum uv.lock)" == "$lock_hash" ]] || die 'Dependency lock changed while the job was running.'
    [[ "$(source_manifest | sha256sum)" == "$source_hash" ]] || die 'Source files changed while the job was running.'
}

run_stage() {
    stage="$1"
    shift
    check_source
    printf '\n[%s] %s\n' "$(date -Is)" "$stage"
    "$@" 2>&1 | tee "$logs/$stage.log"
}

printf 'Run: %s\nCommit: %s\nResults: %s\nMode: %s\n' "$run_id" "$commit" "$results" "$mode"
# No environment dump: credentials such as HF_TOKEN must not enter logs.
run_stage hardware nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
run_stage uv-version uv --version
run_stage install uv sync --locked --python 3.13
# Never run formatters in fix mode or update the dependency lock during a run.
run_stage lint uv run --no-sync ruff check .
run_stage format uv run --no-sync ruff format --check .
run_stage types uv run --no-sync mypy
run_stage tests uv run --no-sync pytest

# Verifies CUDA/bfloat16 and strict loading, then compares exactly 40 QUESTIONS
# against a separate native Transformers load of the pinned official checkpoint.
run_stage reference uv run --no-sync python experiments/qwen3_preflight.py \
    --data "$SWCE_DATA" --results "$run_root/preflight"

check_source
if [[ "$mode" == --preflight-only ]]; then
    printf '\nPreflight-only finished. No smoke, fitting, evaluation, scoring or probes ran.\n'
    printf 'Evidence: %s\n' "$run_root/preflight"
    exit 0
fi

# main.py adds /smoke for --limit; the full run never reads these answers.
# This smoke includes the full method recipes, not a shortened calibration.
run_stage smoke uv run --no-sync python experiments/main.py \
    --model qwen3_vl_8b --limit 16 --results "$results" --data "$SWCE_DATA"

# Preserve the identity of every fitted artifact, including calibration thresholds.
if [[ -d "$SWCE_DATA/artifacts" ]]; then
    find "$SWCE_DATA/artifacts" -type f -exec sha256sum {} + >"$logs/artifacts.sha256"
fi

# One process, one GPU, all three conditions and their controls sequentially.
# Lens is on; residual shift uses 50 images. No limit or bootstrap override.
run_stage full uv run --no-sync python experiments/main.py \
    --model qwen3_vl_8b --jobs baseline vti revis --no-score \
    --results "$results" --data "$SWCE_DATA"
run_stage scoring uv run --no-sync python experiments/main.py \
    --model qwen3_vl_8b --score-only --results "$results" --data "$SWCE_DATA"
run_stage probes uv run --no-sync swce-diagnose --probe \
    --subject swce.models:qwen3_vl_8b \
    methods/vti.py:build methods/vti.py:control methods/revis.py:build methods/revis.py:control \
    --task pope-adversarial --limit 0 --results "$results" --data "$SWCE_DATA"

check_source
printf '\nPipeline finished. Results are NOT approved for publication.\n'
printf 'Review inapplicable methods, REVIS gates/signs, CHAIR loops and shortcuts,\n'
printf 'POPE yes ratio/AUROC/d-prime, MME and baseline literature comparisons.\n'
printf 'Keep the entire run folder (including data/artifacts, preflight and logs): %s\n' "$run_root"
