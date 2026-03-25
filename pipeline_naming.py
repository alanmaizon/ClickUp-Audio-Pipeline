#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_SAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename_stem(value: str) -> str:
    stem = Path(value).stem.strip()
    stem = stem.replace(" ", "_")
    stem = _SAFE_CHAR_RE.sub("_", stem)
    stem = re.sub(r"_+", "_", stem)
    stem = re.sub(r"-+", "-", stem)
    return stem.strip("._-")


def stage_metadata_path(base_dir: Path, stage_dir: str, stem: str) -> Path:
    return base_dir / stage_dir / ".metadata" / f"{stem}.json"


def source_metadata_path(base_dir: Path, stem: str) -> Path:
    return stage_metadata_path(base_dir, "audio-input", stem)


def prepared_metadata_path(base_dir: Path, stem: str) -> Path:
    return stage_metadata_path(base_dir, "audio-prepared", stem)


def output_metadata_path(base_dir: Path, task_id: str) -> Path:
    return base_dir / "audio-output" / ".metadata" / f"{task_id}.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def preferred_stem(*candidates: str, fallback: str = "") -> str:
    for candidate in candidates:
        if not candidate:
            continue
        stem = sanitize_filename_stem(candidate)
        if stem:
            return stem
    return sanitize_filename_stem(fallback) or fallback


def preferred_source_stem(task_id: str, task_name: str, selected_attachment: dict | None = None) -> str:
    selected = selected_attachment or {}
    return preferred_stem(
        str(selected.get("title") or ""),
        str(selected.get("url_filename") or ""),
        task_name,
        fallback=task_id,
    )


def metadata_for_stem(base_dir: Path, stem: str) -> dict:
    for path in (
        prepared_metadata_path(base_dir, stem),
        source_metadata_path(base_dir, stem),
    ):
        payload = load_json(path)
        if payload:
            return payload
    return {}


def resolve_task_id_for_stem(base_dir: Path, stem: str) -> str | None:
    metadata = metadata_for_stem(base_dir, stem)
    task_id = metadata.get("task_id")
    if task_id:
        return str(task_id)
    return None


def write_output_metadata(base_dir: Path, task_id: str, output_filename: str) -> None:
    payload = {
        **metadata_for_stem(base_dir, Path(output_filename).stem),
        "task_id": task_id,
        "output_filename": output_filename,
    }
    path = output_metadata_path(base_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_output_for_upload(base_dir: Path, task_id: str) -> Path | None:
    output_dir = base_dir / "audio-output"
    metadata = load_json(output_metadata_path(base_dir, task_id))
    output_filename = metadata.get("output_filename")
    if output_filename:
        path = output_dir / output_filename
        if path.exists():
            return path

    legacy = output_dir / f"{task_id}.mp3"
    if legacy.exists():
        return legacy
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve Sacred Space audio filenames.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stem_parser = subparsers.add_parser("preferred-source-stem")
    stem_parser.add_argument("--task-id", required=True)
    stem_parser.add_argument("--task-name", default="")
    stem_parser.add_argument("--attachment-title", default="")
    stem_parser.add_argument("--attachment-url-filename", default="")

    task_parser = subparsers.add_parser("task-id-for-stem")
    task_parser.add_argument("--base-dir", type=Path, required=True)
    task_parser.add_argument("--stem", required=True)

    write_parser = subparsers.add_parser("write-output-metadata")
    write_parser.add_argument("--base-dir", type=Path, required=True)
    write_parser.add_argument("--task-id", required=True)
    write_parser.add_argument("--output-filename", required=True)

    resolve_parser = subparsers.add_parser("resolve-output")
    resolve_parser.add_argument("--base-dir", type=Path, required=True)
    resolve_parser.add_argument("--task-id", required=True)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "preferred-source-stem":
        print(
            preferred_source_stem(
                args.task_id,
                args.task_name,
                {
                    "title": args.attachment_title,
                    "url_filename": args.attachment_url_filename,
                },
            )
        )
        return

    if args.command == "task-id-for-stem":
        task_id = resolve_task_id_for_stem(args.base_dir, args.stem)
        if task_id is None:
            raise SystemExit(1)
        print(task_id)
        return

    if args.command == "write-output-metadata":
        write_output_metadata(args.base_dir, args.task_id, args.output_filename)
        return

    resolved = resolve_output_for_upload(args.base_dir, args.task_id)
    if resolved is None:
        raise SystemExit(1)
    print(resolved)


if __name__ == "__main__":
    main()
