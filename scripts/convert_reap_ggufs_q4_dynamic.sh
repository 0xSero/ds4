#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-/home/sero/spark/models}"
HF_CACHE="${HF_CACHE:-$MODEL_ROOT/hf-cache}"
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/ds4-gguf}"
TEMPLATE="${TEMPLATE:-$OUT_DIR/templates/ds4-flash-q2-template-head.gguf}"
THREADS="${THREADS:-8}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

K160_HF="${K160_HF:-}"
K144_HF="${K144_HF:-}"
IMATRIX="${IMATRIX:-}"

find_snapshot() {
    local repo_dir="$1"
    find "$HF_CACHE/$repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1
}

if [[ -z "$K160_HF" ]]; then
    K160_HF="$(find_snapshot "models--0xSero--DeepSeek-V4-Flash-180B-codex-K160-REAP")"
fi
if [[ -z "$K144_HF" ]]; then
    K144_HF="$(find_snapshot "models--0xSero--DeepSeek-V4-Flash-162B-codex-K144-REAP")"
fi

mkdir -p "$OUT_DIR" "$OUT_DIR/status"

if [[ ! -x "$ROOT/gguf-tools/deepseek4-quantize" ]]; then
    make -C "$ROOT/gguf-tools"
fi
if [[ ! -f "$TEMPLATE" ]]; then
    echo "missing template GGUF metadata file: $TEMPLATE" >&2
    exit 2
fi

# Q4-Dynamic is mixed precision in the DS4-compatible sense: routed expert
# tensors use Q4_K, while tensors with fixed DS4 runtime layout requirements
# stay at their template types. Attention projections/shared/output stay Q8_0.
quant_args=(
    --template "$TEMPLATE"
    --experts q4_k
    --attention-proj q8_0
    --shared q8_0
    --output q8_0
    --threads "$THREADS"
)

if [[ -n "$IMATRIX" ]]; then
    quant_args+=(--imatrix "$IMATRIX")
fi

convert_one() {
    local label="$1"
    local hf_dir="$2"
    local out="$3"
    local status="$OUT_DIR/status/$label-Q4-Dynamic.status"
    local partial="$out.partial"

    if [[ ! -f "$hf_dir/config.json" || ! -f "$hf_dir/model.safetensors.index.json" ]]; then
        echo "missing HF snapshot files for $label: $hf_dir" >&2
        exit 3
    fi

    {
        echo "label=$label"
        echo "quant=Q4-Dynamic"
        echo "started=$(date -Is)"
        echo "hf_dir=$hf_dir"
        echo "out=$out"
        echo "template=$TEMPLATE"
        echo "threads=$THREADS"
        echo "dry_run=$DRY_RUN"
        [[ -n "$IMATRIX" ]] && echo "imatrix=$IMATRIX"
    } > "$status"

    if [[ -f "$out" && "$FORCE" != "1" ]]; then
        echo "skipped_existing=$(date -Is)" >> "$status"
        echo "SKIP existing $out"
        return 0
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        "$ROOT/gguf-tools/deepseek4-quantize" \
            --hf "$hf_dir" \
            --dry-run \
            "${quant_args[@]}"
        echo "dry_run_finished=$(date -Is)" >> "$status"
        return 0
    fi

    rm -f "$partial"
    "$ROOT/gguf-tools/deepseek4-quantize" \
        --hf "$hf_dir" \
        --out "$partial" \
        --overwrite \
        "${quant_args[@]}"
    mv "$partial" "$out"
    sha256sum "$out" > "$out.sha256"
    {
        echo "finished=$(date -Is)"
        echo "bytes=$(stat -c '%s' "$out")"
        echo "sha256=$(cut -d' ' -f1 "$out.sha256")"
    } >> "$status"
}

convert_one \
    "DeepSeek-V4-Flash-Spark" \
    "$K160_HF" \
    "$OUT_DIR/DeepSeek-V4-Flash-Spark-Q4-Dynamic-REAP-ds4.gguf"

convert_one \
    "DeepSeek-V4-Flash-Spark-Mini" \
    "$K144_HF" \
    "$OUT_DIR/DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic-REAP-ds4.gguf"
