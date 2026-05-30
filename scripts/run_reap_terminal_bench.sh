#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-DeepSeek-V4-Flash-Spark}"
API_BASE="${API_BASE:-http://127.0.0.1:8010/v1}"
JOBS_DIR="${JOBS_DIR:-/home/sero/spark/benchmarks/terminal-bench-2/jobs}"
HARBOR_HOME="${HARBOR_HOME:-/home/sero/spark/benchmarks/terminal-bench-2}"
HARBOR_VENV="${HARBOR_VENV:-$HARBOR_HOME/.venv}"
AGENT="${AGENT:-terminus-2}"
DATASET="${DATASET:-terminal-bench@2.0}"
N_CONCURRENT="${N_CONCURRENT:-1}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
TIMEOUT_MULTIPLIER="${TIMEOUT_MULTIPLIER:-2.0}"
N_TASKS="${N_TASKS:-}"
TASK_NAME="${TASK_NAME:-}"
DRY_RUN="${DRY_RUN:-0}"
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-placeholder}"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v harbor >/dev/null 2>&1; then
    mkdir -p "$HARBOR_HOME"
    if [[ ! -x "$HARBOR_VENV/bin/harbor" ]]; then
        python3 -m venv "$HARBOR_VENV"
        "$HARBOR_VENV/bin/python" -m pip install --upgrade pip
        "$HARBOR_VENV/bin/python" -m pip install harbor
    fi
    export PATH="$HARBOR_VENV/bin:$PATH"
fi

if ! command -v harbor >/dev/null 2>&1; then
    echo "harbor is not available after install attempt" >&2
    exit 2
fi

mkdir -p "$JOBS_DIR"

export OPENAI_API_KEY
export OPENAI_BASE_URL="$API_BASE"
export OPENAI_API_BASE="$API_BASE"

cmd=(
    harbor run
    --dataset "$DATASET"
    --agent "$AGENT"
    --model "openai/$MODEL"
    --jobs-dir "$JOBS_DIR"
    --n-concurrent "$N_CONCURRENT"
    --n-attempts "$N_ATTEMPTS"
    --timeout-multiplier "$TIMEOUT_MULTIPLIER"
    --yes
    --ae "OPENAI_API_KEY=$OPENAI_API_KEY"
    --ae "OPENAI_BASE_URL=$OPENAI_BASE_URL"
    --ae "OPENAI_API_BASE=$OPENAI_API_BASE"
)

if harbor run --help 2>/dev/null | grep -q -- '--export-traces'; then
    cmd+=(--export-traces)
fi

if [[ -n "$N_TASKS" ]]; then
    cmd+=(--n-tasks "$N_TASKS")
fi
if [[ -n "$TASK_NAME" ]]; then
    cmd+=(--include-task-name "$TASK_NAME")
fi

printf 'Running Terminal-Bench 2.0:'
printf ' %q' "${cmd[@]/$OPENAI_API_KEY/<redacted>}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

exec "${cmd[@]}"
