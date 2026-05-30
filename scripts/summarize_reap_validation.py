#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summarize_variant(path: Path) -> dict:
    bench_rows = []
    bench_csv = path / "ds4-bench.csv"
    if bench_csv.exists():
        with bench_csv.open(newline="", encoding="utf-8") as f:
            bench_rows = list(csv.DictReader(f))

    api = {}
    for probe in sorted(path.glob("api-*.json")):
        data = load_json(probe)
        if not data:
            api[probe.stem] = {"ok": False, "error": "unreadable"}
            continue
        api[probe.stem] = {
            "ok": bool((data.get("eval") or {}).get("passed_core")),
            "ttft_s": data.get("ttft_s"),
            "prefill_tokens_per_s": data.get("prefill_tokens_per_s"),
            "decode_tokens_per_s": data.get("decode_tokens_per_s") or data.get("rough_decode_tokens_per_s"),
            "prompt_tokens": data.get("prompt_tokens"),
            "completion_tokens": data.get("completion_tokens") or data.get("rough_completion_tokens"),
        }

    tb2_results = []
    for result in sorted(path.glob("terminal-bench-2*/jobs/*/result.json")):
        data = load_json(result)
        tb2_results.append(
            {
                "path": str(result),
                "finished_at": (data or {}).get("finished_at"),
                "n_total_trials": (data or {}).get("n_total_trials"),
                "stats": (data or {}).get("stats"),
            }
        )

    return {
        "variant": path.name,
        "model_sha256": (path / "model.sha256").read_text(encoding="utf-8").strip() if (path / "model.sha256").exists() else None,
        "bench_rows": bench_rows,
        "api": api,
        "tb2_results": tb2_results,
        "server_log": str(path / "server.log") if (path / "server.log").exists() else None,
    }


def write_markdown(summary: dict, out: Path) -> None:
    lines = [
        "# DS4 REAP Validation Summary",
        "",
        f"Run directory: `{summary['run_dir']}`",
        "",
    ]
    for variant in summary["variants"]:
        lines.append(f"## {variant['variant']}")
        if variant.get("model_sha256"):
            lines.append("")
            lines.append(f"SHA256: `{variant['model_sha256'].split()[0]}`")
        if variant["bench_rows"]:
            lines.append("")
            lines.append("| ctx | prefill tok/s | decode tok/s | KV bytes |")
            lines.append("| ---: | ---: | ---: | ---: |")
            for row in variant["bench_rows"]:
                lines.append(
                    f"| {row.get('ctx_tokens', '')} | {row.get('prefill_tps', '')} | "
                    f"{row.get('gen_tps', '')} | {row.get('kvcache_bytes', '')} |"
                )
        if variant["api"]:
            lines.append("")
            lines.append("| probe | pass | TTFT s | prefill tok/s | decode tok/s |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for name, probe in sorted(variant["api"].items()):
                lines.append(
                    f"| `{name}` | {probe.get('ok')} | {probe.get('ttft_s')} | "
                    f"{probe.get('prefill_tokens_per_s')} | {probe.get('decode_tokens_per_s')} |"
                )
        if variant["tb2_results"]:
            lines.append("")
            lines.append("Terminal-Bench 2.0 results:")
            for tb2 in variant["tb2_results"]:
                lines.append(f"- `{tb2['path']}` finished_at={tb2.get('finished_at')}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_reap_validation.py RUN_DIR", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    variants = [summarize_variant(path) for path in sorted(run_dir.iterdir()) if path.is_dir()]
    summary = {"run_dir": str(run_dir), "variants": variants}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, run_dir / "SUMMARY.md")
    print(json.dumps({"run_dir": str(run_dir), "variants": [v["variant"] for v in variants]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
