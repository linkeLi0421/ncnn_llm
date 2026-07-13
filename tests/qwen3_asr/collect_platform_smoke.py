#!/usr/bin/env python3
"""Collect platform smoke metadata for Qwen3-ASR local runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def cmd_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_json(path: str | None) -> Any | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def file_info(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": p.stat().st_size,
    }


def fixture_perf(eval_report: Any | None) -> list[dict[str, Any]]:
    if not isinstance(eval_report, dict):
        return []
    out = []
    for fixture in eval_report.get("fixtures", []):
        out.append({
            "id": fixture.get("id"),
            "category": fixture.get("category"),
            "ncnn_pass": fixture.get("ncnn_pass"),
            "chunks": fixture.get("ncnn_chunks"),
            "rtf": fixture.get("ncnn_rtf"),
            "chunking_strategy": fixture.get("chunking_strategy"),
            "module_summary": fixture.get("modules"),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--model")
    parser.add_argument("--eval-report")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out")
    args = parser.parse_args()

    eval_report = load_json(args.eval_report)
    report = {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_brand": cmd_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "cpu_features": cmd_output(["sysctl", "-n", "machdep.cpu.features"]),
            "hw_physicalcpu": cmd_output(["sysctl", "-n", "hw.physicalcpu"]),
            "hw_logicalcpu": cmd_output(["sysctl", "-n", "hw.logicalcpu"]),
        },
        "runtime": {
            "binary": file_info(args.binary),
            "model": file_info(args.model),
            "threads": args.threads,
            "vulkan": False,
            "note": "macOS CPU-only smoke; not a GPU/Vulkan performance result.",
        },
        "evaluation": {
            "eval_report": args.eval_report,
            "summary": eval_report.get("summary") if isinstance(eval_report, dict) else None,
            "fixtures": fixture_perf(eval_report),
        },
    }

    Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.markdown_out:
        lines = [
            "# Qwen3-ASR Platform Smoke",
            "",
            f"- OS: {report['platform']['system']} {report['platform']['release']} ({report['platform']['machine']})",
            f"- CPU: {report['platform']['cpu_brand'] or report['platform']['processor'] or 'unknown'}",
            f"- Threads: {args.threads}",
            f"- Vulkan: false",
            f"- Binary: `{args.binary}`",
            f"- Model: `{args.model}`",
            "",
            "| fixture | pass | chunks | RTF | chunking |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for f in report["evaluation"]["fixtures"]:
            rtf = f.get("rtf")
            rtf_text = f"{rtf:.2f}" if isinstance(rtf, (int, float)) else ""
            lines.append(
                f"| `{f.get('id')}` | {'PASS' if f.get('ncnn_pass') else 'FAIL'} | "
                f"{f.get('chunks') or ''} | {rtf_text} | {f.get('chunking_strategy') or ''} |"
            )
        Path(args.markdown_out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
