#!/bin/sh
set -e

REPO="antirez/deepseek-v4-gguf"
Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
Q2_Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
PRO_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct.gguf"
PRO_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"

# 0xSero REAP DS4 GGUFs live in their own per-model Hugging Face repos.
SPARK_REPO="0xSero/DeepSeek-V4-Flash-Spark-GGUF"
SPARK_FILE="DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf"
SPARK_Q3_FILE="DeepSeek-V4-Flash-Spark-Q3-Dynamic-REAP-ds4.gguf"
SPARK_Q4_FILE="DeepSeek-V4-Flash-Spark-Q4-Dynamic-REAP-ds4.gguf"
SPARK_MINI_REPO="0xSero/DeepSeek-V4-Flash-Spark-Mini-GGUF"
SPARK_MINI_FILE="DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf"
SPARK_MINI_Q3_FILE="DeepSeek-V4-Flash-Spark-Mini-Q3-Dynamic-REAP-ds4.gguf"
SPARK_MINI_Q4_FILE="DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic-REAP-ds4.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DeepSeek V4 GGUF downloader

Usage:
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh q2 [--token TOKEN]
  ./download_model.sh q4 [--token TOKEN]
  ./download_model.sh pro [--token TOKEN]
  ./download_model.sh pro-imatrix [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]
  ./download_model.sh spark [--token TOKEN]          # K160 2-bit (default)
  ./download_model.sh spark-q3 [--token TOKEN]       # K160 3-bit (dynamic)
  ./download_model.sh spark-q4 [--token TOKEN]       # K160 4-bit (dynamic)
  ./download_model.sh spark-mini [--token TOKEN]     # K144 2-bit (default)
  ./download_model.sh spark-mini-q3 [--token TOKEN]  # K144 3-bit (dynamic)
  ./download_model.sh spark-mini-q4 [--token TOKEN]  # K144 4-bit (dynamic)

Targets:
  *** PREFERRED GGUF FILES: USE THE IMATRIX VERSIONS BELOW ***

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  pro-imatrix
       DeepSeek V4 PRO imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines.

  Legacy GGUF files:

  q2   2-bit routed experts, about 81 GB on disk.
       Older non-imatrix model for 96 and 128 GB RAM machines. Prefer
       q2-imatrix unless you specifically need the legacy quant.

  q4   4-bit routed experts, about 153 GB on disk.
       Older non-imatrix model for machines with 256 GB RAM or more. Prefer
       q4-imatrix unless you specifically need the legacy quant.

  pro  DeepSeek V4 PRO non-imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines. Prefer pro-imatrix unless you
       specifically need the legacy quant.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q4-imatrix, q2, and q4, but must be
       enabled explicitly with --mtp when running ds4 or ds4-server.

  0xSero REAP variants (this fork; see REAP.md):

  spark
       DeepSeek V4 Flash Spark, REAP-pruned to 160 routed experts (K160,
       "180B"). Compact Q2 DS4 GGUF, about 81 GB on disk, for 96/128 GB
       machines. From 0xSero/DeepSeek-V4-Flash-Spark-GGUF.

  spark-mini
       DeepSeek V4 Flash Spark Mini, REAP-pruned to 144 routed experts
       (K144, "162B"). Compact Q2 DS4 GGUF. From
       0xSero/DeepSeek-V4-Flash-Spark-Mini-GGUF.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After main-model downloads the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-server --ctx 100000

After downloading mtp, enable it explicitly, for example:
  ./ds4 --mtp <download directory>/$MTP_FILE --mtp-draft 2
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift

# Default repo is antirez's GGUF release; REAP targets override it below.
MODEL_REPO=$REPO

case "$MODEL" in
    q2-imatrix) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    q2-q4-imatrix) MODEL_FILE=$Q2_Q4_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILE=$Q4_IMATRIX_FILE ;;
    q2) MODEL_FILE=$Q2_FILE ;;
    q4) MODEL_FILE=$Q4_FILE ;;
    pro) MODEL_FILE=$PRO_FILE ;;
    pro-imatrix) MODEL_FILE=$PRO_IMATRIX_FILE ;;
    mtp) MODEL_FILE=$MTP_FILE ;;
    spark|spark-q2) MODEL_FILE=$SPARK_FILE; MODEL_REPO=$SPARK_REPO ;;
    spark-q3) MODEL_FILE=$SPARK_Q3_FILE; MODEL_REPO=$SPARK_REPO ;;
    spark-q4) MODEL_FILE=$SPARK_Q4_FILE; MODEL_REPO=$SPARK_REPO ;;
    spark-mini|spark-mini-q2) MODEL_FILE=$SPARK_MINI_FILE; MODEL_REPO=$SPARK_MINI_REPO ;;
    spark-mini-q3) MODEL_FILE=$SPARK_MINI_Q3_FILE; MODEL_REPO=$SPARK_MINI_REPO ;;
    spark-mini-q4) MODEL_FILE=$SPARK_MINI_Q4_FILE; MODEL_REPO=$SPARK_MINI_REPO ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_one() {
    file=$1
    out="$OUT_DIR/$file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$MODEL_REPO/resolve/main/$file"

    mkdir -p "$OUT_DIR"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$MODEL_REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

download_one "$MODEL_FILE"

if [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix, q4-imatrix, q2, and q4."
    echo "Enable it explicitly, for example:"
    echo "  ./ds4 --mtp $OUT_DIR/$MTP_FILE --mtp-draft 2"
else
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
