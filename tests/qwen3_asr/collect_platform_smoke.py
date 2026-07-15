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
        return subprocess.check_output(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def proc_cpuinfo_value(label: str) -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    prefix = label.lower()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == prefix:
            return value.strip()
    return None


def platform_metadata() -> dict[str, Any]:
    system = platform.system()
    cpu_brand = None
    cpu_features = None
    physical_cpu = None
    logical_cpu = str(os.cpu_count()) if os.cpu_count() is not None else None

    if system == "Darwin":
        cpu_brand = cmd_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        cpu_features = cmd_output(["sysctl", "-n", "machdep.cpu.features"])
        physical_cpu = cmd_output(["sysctl", "-n", "hw.physicalcpu"])
        logical_cpu = cmd_output(["sysctl", "-n", "hw.logicalcpu"])
    elif system == "Windows":
        cpu_brand = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or None
    elif system == "Linux":
        cpu_brand = proc_cpuinfo_value("model name") or platform.processor() or None
        cpu_features = proc_cpuinfo_value("flags") or proc_cpuinfo_value("features")

    return {
        "system": system,
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_brand": cpu_brand,
        "cpu_features": cpu_features,
        "hw_physicalcpu": physical_cpu,
        "hw_logicalcpu": logical_cpu,
    }


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


def metrics_for_fixture(fixture: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(fixture, dict):
        return None
    result_path = fixture.get("ncnn_result_json")
    if not result_path:
        return None
    p = Path(result_path)
    metrics_path = p.with_name(p.stem + "_metrics.json")
    metrics = load_json(str(metrics_path))
    if isinstance(metrics, dict):
        return metrics
    return None


def fixture_perf(eval_report: Any | None, fixture_doc: Any | None) -> list[dict[str, Any]]:
    if not isinstance(eval_report, dict):
        return []
    fixture_by_id = {}
    if isinstance(fixture_doc, dict):
        for fixture in fixture_doc.get("fixtures", []):
            if isinstance(fixture, dict):
                fixture_by_id[fixture.get("id")] = fixture
    out = []
    for fixture in eval_report.get("fixtures", []):
        metrics = metrics_for_fixture(fixture_by_id.get(fixture.get("id")))
        out.append({
            "id": fixture.get("id"),
            "category": fixture.get("category"),
            "ncnn_pass": fixture.get("ncnn_pass"),
            "ncnn_semantic_pass": fixture.get("ncnn_semantic_pass"),
            "chunks": fixture.get("ncnn_chunks"),
            "rtf": fixture.get("ncnn_rtf"),
            "max_resident_set_size_bytes": metrics.get("max_resident_set_size_bytes") if metrics else None,
            "chunking_strategy": fixture.get("chunking_strategy"),
            "module_summary": fixture.get("modules"),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--model")
    parser.add_argument("--eval-report")
    parser.add_argument("--fixtures")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out")
    args = parser.parse_args()

    eval_report = load_json(args.eval_report)
    fixture_doc = load_json(args.fixtures)
    report = {
        "schema_version": 1,
        "platform": platform_metadata(),
        "runtime": {
            "binary": file_info(args.binary),
            "model": file_info(args.model),
            "threads": args.threads,
            "vulkan": False,
            "note": f"{platform.system()} CPU-only smoke; not a GPU/Vulkan performance result.",
        },
        "evaluation": {
            "eval_report": args.eval_report,
            "fixtures_doc": args.fixtures,
            "summary": eval_report.get("summary") if isinstance(eval_report, dict) else None,
            "fixtures": fixture_perf(eval_report, fixture_doc),
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
            "| fixture | strict | semantic | chunks | RTF | peak RSS | chunking |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for f in report["evaluation"]["fixtures"]:
            rtf = f.get("rtf")
            rtf_text = f"{rtf:.2f}" if isinstance(rtf, (int, float)) else ""
            rss = f.get("max_resident_set_size_bytes")
            rss_text = f"{rss / 1024 / 1024:.1f} MiB" if isinstance(rss, int) else ""
            lines.append(
                f"| `{f.get('id')}` | {'PASS' if f.get('ncnn_pass') else 'FAIL'} | "
                f"{'PASS' if f.get('ncnn_semantic_pass') else 'FAIL'} | "
                f"{f.get('chunks') or ''} | {rtf_text} | {rss_text} | {f.get('chunking_strategy') or ''} |"
            )
        Path(args.markdown_out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
