#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/sero/spark/models/ds4-gguf}"
OUT_ROOT="${OUT_ROOT:-/home/sero/spark/benchmarks/ds4-reap-gguf}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="$OUT_ROOT/$RUN_ID"
CTX_WINDOWS="${CTX_WINDOWS:-2048 4096 8192 16384 32768 65536 131072 200000}"
CTX_MAX="${CTX_MAX:-200000}"
GEN_TOKENS="${GEN_TOKENS:-128}"
THREADS="${THREADS:-8}"
BASE_PORT="${BASE_PORT:-8010}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
WAIT_FOR_RELEASE="${WAIT_FOR_RELEASE:-0}"
RUN_DS4_BENCH="${RUN_DS4_BENCH:-1}"
RUN_API_PROBES="${RUN_API_PROBES:-1}"
RUN_TB2="${RUN_TB2:-0}"
TB2_VARIANTS="${TB2_VARIANTS:-primary}"
TB2_N_TASKS="${TB2_N_TASKS:-1}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-0}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
DOCKER_HOST_IP="${DOCKER_HOST_IP:-}"

variants=(
    "DeepSeek-V4-Flash-Spark|DeepSeek-V4-Flash-Spark|DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf"
    "DeepSeek-V4-Flash-Spark-Mini|DeepSeek-V4-Flash-Spark-Mini|DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf"
    "DeepSeek-V4-Flash-Spark-Q4-Dynamic|DeepSeek-V4-Flash-Spark|DeepSeek-V4-Flash-Spark-Q4-Dynamic-REAP-ds4.gguf"
    "DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic|DeepSeek-V4-Flash-Spark-Mini|DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic-REAP-ds4.gguf"
)

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/validation.log") 2>&1

echo "started=$(date -Is)"
echo "out_dir=$OUT_DIR"
echo "model_dir=$MODEL_DIR"
echo "ctx_windows=$CTX_WINDOWS"

wait_for_release_pipeline() {
    if [[ "$WAIT_FOR_RELEASE" != "1" ]]; then
        return 0
    fi
    local release_pid_file="/home/sero/spark/logs/ds4-reap-gguf-release.pid"
    while [[ -f "$release_pid_file" ]] && kill -0 "$(cat "$release_pid_file")" 2>/dev/null; do
        echo "waiting_for_release_pid=$(cat "$release_pid_file") date=$(date -Is)"
        tail -n 12 /home/sero/spark/logs/ds4-reap-gguf-release.log 2>/dev/null || true
        sleep 300
    done
}

wait_for_all_files() {
    while true; do
        local missing=0
        for spec in "${variants[@]}"; do
            IFS='|' read -r _label _served file <<< "$spec"
            if [[ ! -f "$MODEL_DIR/$file" || ! -f "$MODEL_DIR/$file.sha256" ]]; then
                missing=1
            fi
        done
        if [[ "$missing" == "0" ]] && ! compgen -G "$MODEL_DIR/*.partial" >/dev/null; then
            return 0
        fi
        echo "waiting_for_ggufs date=$(date -Is)"
        find "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.gguf' -o -name '*.partial' -o -name '*.sha256' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort || true
        sleep 300
    done
}

wait_for_variant_file() {
    local label="$1"
    local file="$2"
    local model_path="$MODEL_DIR/$file"
    local sha_path="$model_path.sha256"
    local partial_path="$model_path.partial"
    while [[ ! -f "$model_path" || ! -f "$sha_path" || -f "$partial_path" ]]; do
        echo "waiting_for_variant=$label date=$(date -Is)"
        find "$MODEL_DIR" -maxdepth 1 -type f \( -name "$file" -o -name "$file.sha256" -o -name "$file.partial" \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort || true
        sleep 300
    done
}

make_prompt_file() {
    local path="$1"
    {
        for i in $(seq 1 26000); do
            printf 'Section %06d: sparse routing prefill decode cache validation repeats with enough neutral text for DS4 context frontier benchmarking.\n' "$i"
        done
    } > "$path"
}

wait_ready() {
    local base_url="$1"
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl -fsS "$base_url/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

docker_host_ip() {
    if [[ -n "$DOCKER_HOST_IP" ]]; then
        printf '%s\n' "$DOCKER_HOST_IP"
        return 0
    fi
    local ip_addr
    ip_addr="$(ip -4 addr show docker0 2>/dev/null | awk '/inet /{sub(/\/.*/, "", $2); print $2; exit}')"
    if [[ -n "$ip_addr" ]]; then
        printf '%s\n' "$ip_addr"
        return 0
    fi
    ip_addr="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -n "$ip_addr" ]]; then
        printf '%s\n' "$ip_addr"
        return 0
    fi
    printf '%s\n' "$CLIENT_HOST"
}

stop_server() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
}

run_variant() {
    local idx="$1"
    local label="$2"
    local served="$3"
    local file="$4"
    local model_path="$MODEL_DIR/$file"
    local variant_dir="$OUT_DIR/$label"
    local port=$((BASE_PORT + idx))
    local base_url="http://$CLIENT_HOST:$port"
    local tb2_base_url="http://$(docker_host_ip):$port/v1"
    local -a ds4_env=()
    mkdir -p "$variant_dir"

    wait_for_variant_file "$label" "$file"
    echo "variant_start=$label date=$(date -Is)"
    sha256sum "$model_path" | tee "$variant_dir/model.sha256"
    ls -lh "$model_path" | tee "$variant_dir/model.ls"
    if [[ "$label" == *Q4-Dynamic* || "$file" == *Q4-Dynamic* ]]; then
        ds4_env+=(DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-64}")
        printf 'DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=%s\n' "${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-64}" | tee "$variant_dir/q4-runtime.env"
    fi

    if [[ "$RUN_DS4_BENCH" == "1" ]]; then
        echo "ds4_bench_start=$label date=$(date -Is)"
        timeout 6h env "${ds4_env[@]}" "$ROOT/ds4-bench" \
            --cuda \
            -m "$model_path" \
            --chat-prompt-file "$OUT_DIR/bench_prompt.txt" \
            --ctx-start 2048 \
            --ctx-max "$CTX_MAX" \
            --ctx-alloc "$((CTX_MAX + GEN_TOKENS + 129))" \
            --step-mul 2 \
            --gen-tokens "$GEN_TOKENS" \
            --threads "$THREADS" \
            --csv "$variant_dir/ds4-bench.csv" \
            > "$variant_dir/ds4-bench.stdout" \
            2> "$variant_dir/ds4-bench.stderr" || echo "ds4_bench_failed=$?"
    fi

    if [[ "$RUN_API_PROBES" != "1" && "$RUN_TB2" != "1" ]]; then
        return 0
    fi

    echo "server_start=$label port=$port date=$(date -Is)"
    env "${ds4_env[@]}" "$ROOT/ds4-server" \
        --cuda \
        --model "$model_path" \
        --ctx "$CTX_MAX" \
        --host "$SERVER_HOST" \
        --port "$port" \
        --threads "$THREADS" \
        --trace "$variant_dir/server.trace" \
        > "$variant_dir/server.log" 2>&1 &
    local server_pid=$!
    echo "$server_pid" > "$variant_dir/server.pid"

    if ! wait_ready "$base_url"; then
        echo "server_ready_failed=$label"
        tail -n 120 "$variant_dir/server.log" || true
        stop_server "$server_pid"
        return 0
    fi

    if [[ "$RUN_API_PROBES" == "1" ]]; then
        for task in smoke code mermaid unicode religion philosophy tool; do
            python3 "$ROOT/scripts/reap_context_probe.py" \
                --base-url "$base_url" \
                --model "$served" \
                --task "$task" \
                --context-tokens 2048 \
                --max-tokens 192 \
                --output "$variant_dir/api-$task.json" || echo "api_probe_failed=$label/$task"
        done
        for ctx in $CTX_WINDOWS; do
            python3 "$ROOT/scripts/reap_context_probe.py" \
                --base-url "$base_url" \
                --model "$served" \
                --task long \
                --context-tokens "$ctx" \
                --max-tokens 64 \
                --output "$variant_dir/api-long-$ctx.json" || echo "api_long_probe_failed=$label/$ctx"
        done
    fi

    if [[ "$RUN_TB2" == "1" && ( "$TB2_VARIANTS" == "all" || "$idx" == "0" ) ]]; then
        echo "tb2_start=$label date=$(date -Is)"
        MODEL="$served" \
        API_BASE="$tb2_base_url" \
        JOBS_DIR="$variant_dir/terminal-bench-2/jobs" \
        N_TASKS="$TB2_N_TASKS" \
        "$ROOT/scripts/run_reap_terminal_bench.sh" > "$variant_dir/tb2.log" 2>&1 || echo "tb2_failed=$label"
    fi

    stop_server "$server_pid"
    echo "variant_finished=$label date=$(date -Is)"
}

wait_for_release_pipeline
make_prompt_file "$OUT_DIR/bench_prompt.txt"

idx=0
for spec in "${variants[@]}"; do
    IFS='|' read -r label served file <<< "$spec"
    run_variant "$idx" "$label" "$served" "$file"
    idx=$((idx + 1))
done

python3 "$ROOT/scripts/summarize_reap_validation.py" "$OUT_DIR" || echo "summary_failed=$?"
if [[ "$UPLOAD_RESULTS" == "1" ]]; then
    python3 "$ROOT/scripts/upload_reap_validation_results.py" "$OUT_DIR" || echo "upload_results_failed=$?"
fi
echo "finished=$(date -Is)"
