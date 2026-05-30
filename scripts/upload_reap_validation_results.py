#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi


REPOS = {
    "DeepSeek-V4-Flash-Spark": {
        "repo_id": "0xSero/DeepSeek-V4-Flash-Spark-GGUF",
        "source_model": "0xSero/DeepSeek-V4-Flash-180B",
    },
    "DeepSeek-V4-Flash-Spark-Q4-Dynamic": {
        "repo_id": "0xSero/DeepSeek-V4-Flash-Spark-GGUF",
        "source_model": "0xSero/DeepSeek-V4-Flash-180B",
    },
    "DeepSeek-V4-Flash-Spark-Mini": {
        "repo_id": "0xSero/DeepSeek-V4-Flash-Spark-Mini-GGUF",
        "source_model": "0xSero/DeepSeek-V4-Flash-162B",
    },
    "DeepSeek-V4-Flash-Spark-Mini-Q4-Dynamic": {
        "repo_id": "0xSero/DeepSeek-V4-Flash-Spark-Mini-GGUF",
        "source_model": "0xSero/DeepSeek-V4-Flash-162B",
    },
}

TOP_LEVEL_FILES = {"SUMMARY.md", "summary.json", "validation.log"}
SKIP_NAMES = {"bench_prompt.txt", "server.pid", "server.trace"}
SKIP_SUFFIXES = {".gguf", ".partial"}


def should_copy(path: Path, max_bytes: int) -> bool:
    if not path.is_file():
        return False
    if path.name in SKIP_NAMES:
        return False
    if any(path.name.endswith(suffix) for suffix in SKIP_SUFFIXES):
        return False
    try:
        return path.stat().st_size <= max_bytes
    except OSError:
        return False


def copy_tree(src: Path, dst: Path, max_bytes: int) -> None:
    for path in src.rglob("*"):
        if not should_copy(path, max_bytes):
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def write_results_readme(staging: Path, repo_id: str, source_model: str, run_dir: Path, variants: list[str]) -> None:
    variant_rows = "\n".join(f"- `{variant}`" for variant in sorted(variants))
    (staging / "README.md").write_text(
        f"""# Validation Results

Run directory on Spark: `{run_dir}`

Source Spark model: https://huggingface.co/{source_model}

GGUF repo: https://huggingface.co/{repo_id}

Included variants:

{variant_rows}

Artifacts in this folder are benchmark and validation evidence for the DS4/DwarfStar GGUF release. They include DS4 context-window throughput CSVs, OpenAI-compatible API probes, long-context needle checks, and Terminal-Bench 2.0 logs when that stage was enabled.
""",
        encoding="utf-8",
    )


def stage_for_repo(results_dir: Path, staging_root: Path, repo_id: str, source_model: str, variants: list[str], max_bytes: int) -> Path:
    run_id = results_dir.name
    staging = staging_root / repo_id.replace("/", "__") / run_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for name in TOP_LEVEL_FILES:
        src = results_dir / name
        if src.is_file() and should_copy(src, max_bytes):
            shutil.copy2(src, staging / name)

    for variant in variants:
        src = results_dir / variant
        if src.is_dir():
            copy_tree(src, staging / variant, max_bytes)

    write_results_readme(staging, repo_id, source_model, results_dir, variants)
    return staging


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload DS4 REAP validation results to the matching GGUF model repos.")
    parser.add_argument("results_dir", help="Validation run directory, e.g. /home/sero/spark/benchmarks/ds4-reap-gguf/20260528T154500Z")
    parser.add_argument("--staging-root", default="/home/sero/spark/benchmarks/ds4-reap-gguf/upload-results-staging")
    parser.add_argument("--max-file-mib", type=int, default=64, help="Skip individual result files larger than this.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"missing results directory: {results_dir}")

    by_repo: dict[str, dict] = {}
    for variant, meta in REPOS.items():
        if (results_dir / variant).is_dir():
            entry = by_repo.setdefault(
                meta["repo_id"],
                {"source_model": meta["source_model"], "variants": []},
            )
            entry["variants"].append(variant)

    if not by_repo:
        raise FileNotFoundError(f"no recognized variant result directories found under {results_dir}")

    api = HfApi()
    max_bytes = args.max_file_mib * 1024 * 1024
    staging_root = Path(args.staging_root)
    run_id = results_dir.name

    for repo_id, meta in sorted(by_repo.items()):
        staging = stage_for_repo(
            results_dir=results_dir,
            staging_root=staging_root,
            repo_id=repo_id,
            source_model=meta["source_model"],
            variants=meta["variants"],
            max_bytes=max_bytes,
        )
        print(f"staged validation results for {repo_id}: {staging}", flush=True)
        if args.dry_run:
            continue
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(staging),
            path_in_repo=f"validation/{run_id}",
        )
        print(f"uploaded validation/{run_id} to {repo_id}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
