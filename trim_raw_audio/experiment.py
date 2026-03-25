from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import TrimConfig
from .pipeline import METHODS, RawAudioPreparer


def _timestamp_label() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _numeric_values(entries: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_experiment(
    preparer: RawAudioPreparer,
    *,
    config: TrimConfig,
    input_dir: Path,
    artifact_root: Path,
    methods: list[str],
    file_names: list[str] | None = None,
    sample_size: int | None = None,
    dry_run: bool = False,
    debug: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    invalid_methods = [method for method in methods if method not in METHODS]
    if invalid_methods:
        raise ValueError(f"Unsupported experiment method(s): {', '.join(invalid_methods)}")

    all_inputs = preparer.collect_inputs(input_dir, file_names=file_names)
    if sample_size is not None:
        all_inputs = all_inputs[:sample_size]
    selected_names = [path.name for path in all_inputs]

    run_id = _timestamp_label()
    run_root = artifact_root / run_id
    per_method_rows: list[dict[str, Any]] = []
    method_summaries: list[dict[str, Any]] = []

    for method in methods:
        method_root = run_root / method
        output_dir = method_root / "audio"
        artifacts_dir = method_root / "artifacts"
        result = preparer.prepare_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            artifact_root=artifacts_dir,
            requested_method=method,
            file_names=selected_names,
            dry_run=dry_run,
            debug=debug,
            force=force,
            skip_existing=False,
            run_label=f"experiment:{run_id}:{method}",
        )
        for row in result["rows"]:
            enriched = dict(row)
            enriched["experiment_method"] = method
            per_method_rows.append(enriched)
        method_summaries.append(result["summary"])

    comparison: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in per_method_rows:
        comparison[row["file_id"]][row["experiment_method"]] = row

    ranked_files: list[dict[str, Any]] = []
    for file_id, method_rows in comparison.items():
        method_entries = list(method_rows.values())
        start_offsets = _numeric_values(method_entries, "final_start_offset_sec")
        end_offsets = _numeric_values(method_entries, "final_end_offset_sec")
        confidences = _numeric_values(method_entries, "confidence_score")
        max_start_delta = round(max(start_offsets) - min(start_offsets), 3) if start_offsets else 0.0
        max_end_delta = round(max(end_offsets) - min(end_offsets), 3) if end_offsets else 0.0
        failure_penalty = sum(1.0 for entry in method_entries if entry.get("method_chosen") == "error")
        uncertainty = round((1.0 - min(confidences or [0.0])) + max_start_delta + max_end_delta + failure_penalty, 3)
        review_needed = any(entry.get("manual_review") or entry.get("method_chosen") == "error" for entry in method_entries)
        ranked_files.append(
            {
                "file_id": file_id,
                "needs_manual_review": review_needed,
                "uncertainty_score": uncertainty,
                "max_start_delta_sec": max_start_delta,
                "max_end_delta_sec": max_end_delta,
                "methods": method_rows,
            }
        )

    ranked_files.sort(
        key=lambda item: (
            not item["needs_manual_review"],
            -item["uncertainty_score"],
            -item["max_start_delta_sec"],
        )
    )

    shortlist = ranked_files[: config.experiment.shortlist_size]
    shortlist_path = run_root / "manual-review-shortlist.json"
    shortlist_path.write_text(json.dumps(shortlist, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        f"# Raw Trim Experiment {run_id}",
        "",
        f"Input directory: `{input_dir}`",
        f"Methods: {', '.join(methods)}",
        f"Dry run: `{dry_run}`",
        "",
        "## Manual Review Shortlist",
        "",
    ]
    if not shortlist:
        report_lines.append("No files were selected for manual review.")
    else:
        for item in shortlist:
            report_lines.append(f"### {item['file_id']}")
            report_lines.append(f"- Needs manual review: `{item['needs_manual_review']}`")
            report_lines.append(f"- Uncertainty score: `{item['uncertainty_score']}`")
            report_lines.append(f"- Max start delta: `{item['max_start_delta_sec']}` sec")
            report_lines.append(f"- Max end delta: `{item['max_end_delta_sec']}` sec")
            for method in methods:
                entry = item["methods"].get(method)
                if not entry:
                    continue
                if entry.get("method_chosen") == "error":
                    report_lines.append(
                        f"- `{method}` failed: `{entry.get('failure_or_fallback_reason') or 'unknown error'}`"
                    )
                else:
                    report_lines.append(
                        f"- `{method}`: start `{entry['final_start_offset_sec']}` sec, "
                        f"end `{entry['final_end_offset_sec']}` sec, confidence `{entry['confidence_score']}`, "
                        f"trimmed `{entry['trimmed_path'] or 'dry-run only'}`"
                    )
                if entry.get("predicted_intro_end_sec") not in ("", None):
                    report_lines.append(f"- `{method}` predicted intro end: `{entry['predicted_intro_end_sec']}` sec")
                if entry.get("transcript_snippet"):
                    report_lines.append(f"- `{method}` transcript: {entry['transcript_snippet']}")
            report_lines.append("")

    report_path = run_root / "report.md"
    _write_markdown(report_path, "\n".join(report_lines).rstrip() + "\n")

    summary = {
        "run_id": run_id,
        "run_root": str(run_root),
        "selected_files": selected_names,
        "method_summaries": method_summaries,
        "shortlist_json": str(shortlist_path),
        "report_md": str(report_path),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
