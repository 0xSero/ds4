#!/usr/bin/env bash
#
# deploy_spark.sh — one command to run REAP DeepSeek V4 Flash on a DGX Spark.
#
# Builds the DwarfStar CUDA engine, downloads a prebuilt DS4 GGUF, and serves it
# over the OpenAI-compatible HTTP API at maximum context and speed, tuned for the
# Spark's unified-memory budget. Models larger than RAM are streamed straight
# from NVMe by the engine's host-mapped / direct-IO loader (see ds4_cuda.cu), so
# the 82 GB "Spark" (180B/K160) Q3 checkpoint runs even on a 64 GB box.
#
# Everything is idempotent: a present CUDA toolkit, an up-to-date build, and an
# already-downloaded GGUF are all detected and skipped, so re-running is cheap.
#
# Usage:
#   scripts/deploy_spark.sh                      # 180B/K160 Q3-Dynamic on :8000
#   MODEL=spark-mini-q3 scripts/deploy_spark.sh  # 162B/K144 Q3-Dynamic
#   CTX=393216 scripts/deploy_spark.sh           # bigger context (slower; see note)
#   PORT=8000 HOST=0.0.0.0 scripts/deploy_spark.sh
#   SERVE=0 scripts/deploy_spark.sh              # build + download only
#   SERVE_FOREGROUND=1 scripts/deploy_spark.sh   # run server in foreground
#
# Context vs speed: this box's unified memory is shared by the model (streamed),
# the KV cache (~44 KB/token for the MLA stack) and activations. Larger CTX means
# more KV and less room to keep experts resident, so decode slows. CTX=131072
# (128K) is a good balance on a 64 GB Spark; raise it if you need more context.
#
set -euo pipefail

log()  { printf '\033[1;36m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy:warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[deploy:error]\033[0m %s\n' "$*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ----------------------------- configuration ------------------------------
MODEL="${MODEL:-spark-q3}"                       # download_model.sh target
DS4_GGUF_DIR="${DS4_GGUF_DIR:-$ROOT/gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX="${CTX:-131072}"                             # startup KV context (128K)
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
SERVE="${SERVE:-1}"
SERVE_FOREGROUND="${SERVE_FOREGROUND:-0}"
HF_TOKEN="${HF_TOKEN:-}"
CUDA_PKG="${CUDA_PKG:-cuda-toolkit-13-0}"
CUDA_REPO="${CUDA_REPO:-ubuntu2404/sbsa}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
export DS4_GGUF_DIR

# Preferred -> fallback download targets (first that fits the disk wins).
FALLBACK_CHAIN="${FALLBACK_CHAIN:-spark-q3 spark-mini-q3 spark-q2 spark-mini-q2}"

# ------------------------------ 1. CUDA -----------------------------------
find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return 0; fi
    local d
    for d in /usr/local/cuda /usr/local/cuda-*; do
        [ -x "$d/bin/nvcc" ] && { echo "$d/bin/nvcc"; return 0; }
    done
    return 1
}

ensure_cuda() {
    local nvcc
    if ! nvcc="$(find_nvcc)"; then
        log "no nvcc found — installing $CUDA_PKG from $CUDA_REPO"
        [ "$(id -u)" = 0 ] || die "CUDA toolkit missing and not running as root; install $CUDA_PKG manually"
        export DEBIAN_FRONTEND=noninteractive
        local kr=/tmp/cuda-keyring.deb
        wget -qO "$kr" "https://developer.download.nvidia.com/compute/cuda/repos/$CUDA_REPO/cuda-keyring_1.1-1_all.deb" \
            || die "failed to fetch cuda-keyring"
        dpkg -i "$kr" >/dev/null 2>&1 || die "failed to install cuda-keyring"
        apt-get update -y >/dev/null || die "apt-get update failed"
        apt-get install -y "$CUDA_PKG" >/dev/null || die "apt-get install $CUDA_PKG failed"
        nvcc="$(find_nvcc)" || die "nvcc still not found after install"
    fi
    CUDA_HOME="$(dirname "$(dirname "$nvcc")")"
    export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
    # Make the CUDA runtime libs resolvable at run time even if ldconfig wasn't
    # configured for the sbsa target path.
    export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    log "CUDA toolkit at $CUDA_HOME ($("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1))"
}

# ------------------------------ 2. build ----------------------------------
build_engine() {
    # The repo ships prebuilt object files; clean so the Spark recompiles the
    # CUDA translation unit for its own GB10 arch instead of relinking stale ones.
    log "building DwarfStar for GB10 (make clean && make cuda-spark)"
    make clean >/dev/null 2>&1 || true
    make cuda-spark
    [ -x "$ROOT/ds4-server" ] && [ -x "$ROOT/ds4" ] || die "build did not produce ds4 / ds4-server"
    log "build complete"
}

# -------------------------- 3. model download -----------------------------
disk_free_gb() {
    df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'
}

# Approximate on-disk sizes (GB) for the fallback chain.
model_size_gb() {
    case "$1" in
        spark-q3)       echo 86 ;;
        spark-mini-q3)  echo 79 ;;
        spark-q2)       echo 61 ;;
        spark-mini-q2)  echo 56 ;;
        spark-q4)       echo 100 ;;
        spark-mini-q4)  echo 100 ;;
        *)              echo 90 ;;
    esac
}

# GGUF filename for a download target (mirrors download_model.sh).
model_file() {
    case "$1" in
        spark-q3)       echo "DeepSeek-V4-Flash-Spark-Q3-Dynamic-REAP-ds4.gguf" ;;
        spark-mini-q3)  echo "DeepSeek-V4-Flash-Spark-Mini-Q3-Dynamic-REAP-ds4.gguf" ;;
        spark-q2)       echo "DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf" ;;
        spark-mini-q2)  echo "DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf" ;;
        spark-q4)       echo "DeepSeek-V4-Flash-Spark-Q4-Dynamic-REAP-ds4.gguf" ;;
        spark-mini-q4)  echo "DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic-REAP-ds4.gguf" ;;
        *)              echo "" ;;
    esac
}

pick_model() {
    mkdir -p "$DS4_GGUF_DIR"
    local free cand f need
    free="$(disk_free_gb "$DS4_GGUF_DIR")"
    # 1) Honor an already-downloaded checkpoint (preferred MODEL first), so a
    #    re-run never re-picks just because the file itself consumed the disk.
    for cand in "$MODEL" $FALLBACK_CHAIN; do
        f="$(model_file "$cand")"
        if [ -n "$f" ] && [ -s "$DS4_GGUF_DIR/$f" ]; then echo "$cand"; return 0; fi
    done
    # 2) Otherwise pick the first candidate that fits the free disk (MODEL first).
    for cand in "$MODEL" $FALLBACK_CHAIN; do
        need="$(model_size_gb "$cand")"
        if [ "$free" -ge "$need" ]; then
            [ "$cand" = "$MODEL" ] || warn "requested '$MODEL' will not fit in ${free}G free; using '$cand' instead"
            echo "$cand"; return 0
        fi
    done
    die "no checkpoint fits in ${free}G free disk; free space or set DS4_GGUF_DIR to a bigger volume"
}

download_model() {
    MODEL="$(pick_model)"
    log "fetching model target: $MODEL (into $DS4_GGUF_DIR)"
    # download_model.sh resumes a partial via curl -C -, so retry on a dropped
    # connection until the (idempotent) fetch reports success.
    local try
    for try in 1 2 3 4 5 6 7 8; do
        if [ -n "$HF_TOKEN" ]; then
            ./download_model.sh "$MODEL" --token "$HF_TOKEN" && break
        else
            ./download_model.sh "$MODEL" && break
        fi
        warn "download attempt $try interrupted; resuming in 5s…"; sleep 5
    done
    [ -e "$ROOT/ds4flash.gguf" ] || die "ds4flash.gguf symlink missing after download"
    local target; target="$(readlink -f "$ROOT/ds4flash.gguf" || true)"
    [ -s "$target" ] || die "model file empty/missing: $target"
    local sha="$DS4_GGUF_DIR/$(basename "$target").sha256"
    if [ -s "$sha" ]; then
        log "verifying sha256…"
        ( cd "$DS4_GGUF_DIR" && sha256sum -c "$(basename "$sha")" ) || die "sha256 mismatch for $target"
    fi
    log "model ready: $target ($(du -h "$target" | cut -f1))"
}

# ----------------- 4. memory-aware streaming env --------------------------
tune_env() {
    local mem_gb; mem_gb="$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)"
    if [ -z "${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-}" ]; then
        # Estimate the KV-cache footprint (~44 KB/token for this MLA model) plus
        # headroom for activations and the OS; the engine streams whatever does
        # not fit straight from the GGUF on NVMe.
        local kv_gb=$(( (CTX * 44 / 1000000) + 1 ))
        local cap=$(( mem_gb - kv_gb - 6 ))
        [ "$cap" -lt 8 ] && cap=8
        export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="$cap"
    fi
    export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB
    log "unified ${mem_gb}G, ctx ${CTX} -> DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=$DS4_CUDA_WEIGHT_CACHE_LIMIT_GB (overflow streams from NVMe)"
}

# ------------------------------ 5. serve ----------------------------------
server_alive() { pgrep -f "ds4-server .*--port $PORT" >/dev/null 2>&1; }

serve() {
    local model_path; model_path="$(readlink -f "$ROOT/ds4flash.gguf")"
    local -a args=(--cuda -m "$model_path" --ctx "$CTX" --host "$HOST" --port "$PORT" --threads "$THREADS")
    log "starting: ds4-server ${args[*]}"
    if [ "$SERVE_FOREGROUND" = 1 ]; then exec "$ROOT/ds4-server" "${args[@]}"; fi

    setsid "$ROOT/ds4-server" "${args[@]}" > "$ROOT/ds4-server.log" 2>&1 < /dev/null &
    echo $! > "$ROOT/ds4-server.pid"
    log "ds4-server starting; waiting for /v1/models (log: $ROOT/ds4-server.log)"
    local url="http://127.0.0.1:$PORT" deadline=$((SECONDS + READY_TIMEOUT))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -fsS "$url/v1/models" >/dev/null 2>&1; then
            local id; id="$(curl -fsS "$url/v1/models" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)"
            log "READY — OpenAI-compatible API at http://$HOST:$PORT/v1 (model id: ${id:-unknown})"
            log "  test: curl http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}'"
            return 0
        fi
        server_alive || { warn "ds4-server exited during startup:"; tail -n 60 "$ROOT/ds4-server.log" >&2; die "server failed to start"; }
        sleep 5
    done
    die "server not ready within ${READY_TIMEOUT}s; see $ROOT/ds4-server.log"
}

# ------------------------------- main -------------------------------------
log "DGX Spark deploy: MODEL=$MODEL CTX=$CTX PORT=$PORT"
ensure_cuda
build_engine
download_model
tune_env
if [ "$SERVE" = 1 ]; then
    serve
else
    log "build + download complete (SERVE=$SERVE); start manually with:"
    log "  DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=$DS4_CUDA_WEIGHT_CACHE_LIMIT_GB ./ds4-server --cuda -m ds4flash.gguf --ctx $CTX --host $HOST --port $PORT"
fi
