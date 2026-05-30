#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-/home/sero/spark/models}"
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/ds4-gguf}"
Q2_PID_FILE="${Q2_PID_FILE:-/home/sero/spark/logs/ds4-reap-gguf-convert.pid}"
Q2_LOG="${Q2_LOG:-/home/sero/spark/logs/ds4-reap-gguf-convert.log}"

wait_for_q2() {
    if [[ ! -f "$Q2_PID_FILE" ]]; then
        echo "missing Q2 pid file: $Q2_PID_FILE" >&2
        return 1
    fi
    local pid
    pid="$(cat "$Q2_PID_FILE")"
    echo "waiting_for_q2_pid=$pid"
    while kill -0 "$pid" 2>/dev/null; do
        date -Is
        tail -n 8 "$Q2_LOG" 2>/dev/null || true
        sleep 300
    done

    if ! grep -q ' rc=0$' "$Q2_LOG"; then
        echo "Q2 conversion did not finish cleanly; refusing to build/upload follow-up artifacts." >&2
        tail -n 80 "$Q2_LOG" >&2 || true
        return 1
    fi
}

wait_for_q2

cd "$ROOT"
THREADS="${THREADS:-8}" FORCE="${FORCE:-0}" ./scripts/convert_reap_ggufs_q4_dynamic.sh
python3 ./scripts/upload_reap_ggufs.py

{
    echo "release_finished=$(date -Is)"
    ls -lh "$OUT_DIR"/DeepSeek-V4-Flash-Spark*REAP-ds4.gguf "$OUT_DIR"/DeepSeek-V4-Flash-Spark*REAP-ds4.gguf.sha256
} > "$OUT_DIR/status/release-upload.status"
