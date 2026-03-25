from __future__ import annotations

import json
from pathlib import Path

from pipeline_naming import preferred_source_stem, resolve_output_for_upload, resolve_task_id_for_stem, write_output_metadata


def test_preferred_source_stem_prefers_attachment_title() -> None:
    stem = preferred_source_stem(
        "task001",
        "wrong-task-name",
        {
            "title": "correct-audio-name.wav",
            "url_filename": "url-fallback.wav",
        },
    )
    assert stem == "correct-audio-name"


def test_preferred_source_stem_falls_back_to_url_filename() -> None:
    stem = preferred_source_stem(
        "task001",
        "some-task-name",
        {
            "title": "",
            "url_filename": "url-audio-file.wav",
        },
    )
    assert stem == "url-audio-file"


def test_preferred_source_stem_falls_back_to_task_name() -> None:
    stem = preferred_source_stem(
        "task001",
        "task-name-fallback",
        {
            "title": "",
            "url_filename": "",
        },
    )
    assert stem == "task-name-fallback"


def test_preferred_source_stem_falls_back_to_task_id() -> None:
    stem = preferred_source_stem("task001", "", {})
    assert stem == "task001"


def test_preferred_source_stem_no_attachment() -> None:
    stem = preferred_source_stem("task001", "my-task", None)
    assert stem == "my-task"


def test_resolve_task_id_for_stem_reads_prepared_metadata(tmp_path: Path) -> None:
    meta_path = tmp_path / "audio-prepared" / ".metadata" / "test-audio.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"task_id": "task001", "prepared_filename": "test-audio.wav"}),
        encoding="utf-8",
    )
    resolved = resolve_task_id_for_stem(tmp_path, "test-audio")
    assert resolved == "task001"


def test_output_metadata_resolves_human_readable_mp3_for_upload(tmp_path: Path) -> None:
    source_meta = tmp_path / "audio-prepared" / ".metadata" / "test-audio.json"
    source_meta.parent.mkdir(parents=True, exist_ok=True)
    source_meta.write_text(json.dumps({"task_id": "task001"}), encoding="utf-8")

    output_path = tmp_path / "audio-output" / "test-audio.mp3"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"mp3")

    write_output_metadata(tmp_path, "task001", output_path.name)
    resolved = resolve_output_for_upload(tmp_path, "task001")
    assert resolved == output_path
