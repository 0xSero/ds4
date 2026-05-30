#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


ARTIFACTS = [
    {
        "label": "DeepSeek-V4-Flash-Spark",
        "repo_id": "0xSero/DeepSeek-V4-Flash-Spark-GGUF",
        "source_model": "0xSero/DeepSeek-V4-Flash-180B",
        "conversion_source": "0xSero/DeepSeek-V4-Flash-180B-codex-K160-REAP",
        "files": [
            "DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf",
            "DeepSeek-V4-Flash-Spark-Q2-REAP-ds4.gguf.sha256",
            "DeepSeek-V4-Flash-Spark-Q4-Dynamic-REAP-ds4.gguf",
            "DeepSeek-V4-Flash-Spark-Q4-Dynamic-REAP-ds4.gguf.sha256",
        ],
    },
    {
        "label": "DeepSeek-V4-Flash-Spark-Mini",
        "repo_id": "0xSero/DeepSeek-V4-Flash-Spark-Mini-GGUF",
        "source_model": "0xSero/DeepSeek-V4-Flash-162B",
        "conversion_source": "0xSero/DeepSeek-V4-Flash-162B-codex-K144-REAP",
        "files": [
            "DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf",
            "DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf.sha256",
            "DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic-REAP-ds4.gguf",
            "DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic-REAP-ds4.gguf.sha256",
        ],
    },
]


def read_sha(path: Path) -> str:
    return path.read_text(encoding="utf-8").split()[0]


def write_card(dst: Path, item: dict, out_dir: Path) -> None:
    rows = []
    for name in item["files"]:
        if name.endswith(".sha256"):
            continue
        sha_path = out_dir / f"{name}.sha256"
        size_gib = (out_dir / name).stat().st_size / (1024 ** 3)
        rows.append(f"| `{name}` | {size_gib:.2f} GiB | `{read_sha(sha_path)}` |")

    dst.write_text(
        f"""---
base_model:
- {item["source_model"]}
library_name: gguf
tags:
- deepseek-v4
- gguf
- dgx-spark
- reap
- dwarfstar
- ds4
---

# {item["label"]} GGUF

This repository contains DS4/DwarfStar GGUF conversions of `{item["label"]}`.

The GGUFs point back to the original Spark Hugging Face model:

- Original Spark model: https://huggingface.co/{item["source_model"]}
- Conversion source checkpoint: https://huggingface.co/{item["conversion_source"]}
- Runtime/converter repo: https://github.com/antirez/ds4
- Spark deployment repo: https://github.com/0xSero/deepseek-spark

## Files

| File | Size | SHA256 |
| --- | ---: | --- |
{os.linesep.join(rows)}

## Quantization

- `Q2-REAP-ds4`: compact DS4 profile using `IQ2_XXS` routed gate/up experts, `Q2_K` routed down experts, and `Q8_0` shared/output/attention projections.
- `Q4-Dynamic-REAP-ds4`: quality-biased DS4 profile using `Q4_K` routed experts while preserving DS4 fixed-layout tensors at their runtime-required types; attention projections, shared experts, and output tensors use `Q8_0`.

These are DS4/DwarfStar-specific GGUF files for DeepSeek-V4 Flash REAP checkpoints. They are not generic llama.cpp files unless your runtime supports the same DeepSeek-V4 Flash tensor layout and DS4 metadata.

## Validation

Context-window, API, tool-call, and Terminal-Bench 2.0 artifacts are uploaded under `validation/<run-id>/` after the Spark validation suite finishes.
""",
        encoding="utf-8",
    )


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_item(out_dir: Path, staging_root: Path, item: dict) -> Path:
    missing = [name for name in item["files"] if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{item['label']} missing required files: {', '.join(missing)}")

    staging = staging_root / item["repo_id"].replace("/", "__")
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.iterdir():
        if old.is_file():
            old.unlink()

    for name in item["files"]:
        hardlink_or_copy(out_dir / name, staging / name)
    write_card(staging / "README.md", item, out_dir)
    return staging


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload DeepSeek-V4-Flash Spark DS4 GGUF releases.")
    parser.add_argument("--out-dir", default="/home/sero/spark/models/ds4-gguf")
    parser.add_argument("--staging-root", default="/home/sero/spark/models/ds4-gguf/upload-staging")
    parser.add_argument("--private", action="store_true", help="Create/update private repos instead of public repos.")
    parser.add_argument("--dry-run", action="store_true", help="Stage files and print intended uploads without uploading.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    staging_root = Path(args.staging_root)
    api = HfApi()

    for item in ARTIFACTS:
        staging = stage_item(out_dir, staging_root, item)
        print(f"staged {item['repo_id']} at {staging}", flush=True)
        if args.dry_run:
            continue
        api.create_repo(repo_id=item["repo_id"], repo_type="model", private=args.private, exist_ok=True)
        api.upload_large_folder(
            repo_id=item["repo_id"],
            repo_type="model",
            folder_path=str(staging),
        )
        print(f"uploaded {item['repo_id']}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
