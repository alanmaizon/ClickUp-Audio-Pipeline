from __future__ import annotations

import json
from unittest.mock import patch

import workflow1_download as download_module
from workflow1_download import (
    _candidate_audio_extension,
    _check_name_mismatch,
    _resolve_download_path,
    select_audio_attachment,
)


def test_candidate_audio_extension_from_url_when_title_has_no_suffix() -> None:
    ext = _candidate_audio_extension(
        "recording-without-extension",
        "https://example.com/uploads/recording.m4a?download=1",
    )
    assert ext == ".m4a"


def test_candidate_audio_extension_from_title() -> None:
    ext = _candidate_audio_extension(
        "recording.wav",
        "https://example.com/uploads/blob",
    )
    assert ext == ".wav"


def test_candidate_audio_extension_returns_none_for_non_audio() -> None:
    ext = _candidate_audio_extension("notes.txt", "https://example.com/notes.txt")
    assert ext is None


def test_select_audio_attachment_prefers_non_copy() -> None:
    attachments = [
        {
            "id": "copy",
            "title": "recording-copy.m4a",
            "url": "https://example.com/copy.m4a",
            "date": "200",
        },
        {
            "id": "primary",
            "title": "recording.m4a",
            "url": "https://example.com/primary.m4a",
            "date": "100",
        },
    ]
    selected = select_audio_attachment(attachments)
    assert selected is not None
    assert selected["id"] == "primary"


def test_resolve_download_path_uses_attachment_stem(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(download_module, "INPUT_DIR", tmp_path)
    monkeypatch.setattr(download_module, "INPUT_META_DIR", tmp_path / ".metadata")

    task = {"name": "wrong-task-name"}
    attachment = {
        "title": "correct-audio-name.wav",
        "url": "https://example.com/correct-audio-name.wav",
    }
    dest_path = _resolve_download_path(task, "task001", attachment)
    assert dest_path.name == "correct-audio-name.wav"


def test_resolve_download_path_falls_back_to_task_name(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(download_module, "INPUT_DIR", tmp_path)
    monkeypatch.setattr(download_module, "INPUT_META_DIR", tmp_path / ".metadata")

    task = {"name": "task-name-fallback"}
    attachment = {
        "title": "",
        "url": "",
    }
    dest_path = _resolve_download_path(task, "task001", attachment)
    assert dest_path.stem == "task-name-fallback"


# ── Duplicate name detection ─────────────────────────────────────────────────


def test_resolve_download_path_appends_task_id_on_collision(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(download_module, "INPUT_DIR", tmp_path)
    meta_dir = tmp_path / ".metadata"
    monkeypatch.setattr(download_module, "INPUT_META_DIR", meta_dir)
    meta_dir.mkdir()

    # First task claims the preferred stem
    first_meta = meta_dir / "recording.json"
    first_meta.write_text('{"task_id": "task001"}', encoding="utf-8")
    (tmp_path / "recording.wav").write_bytes(b"audio")

    # Second task has the same attachment filename
    task = {"name": "recording"}
    attachment = {"title": "recording.wav", "url": "https://example.com/recording.wav"}

    with patch.object(download_module, "api_post") as mock_post:
        dest_path = _resolve_download_path(task, "task002", attachment)

    assert "task002" in dest_path.stem
    assert dest_path.stem != "recording"
    mock_post.assert_called_once()
    body = mock_post.call_args[0][1]
    assert "Duplicate" in body["comment_text"]


def test_resolve_download_path_no_note_when_no_collision(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(download_module, "INPUT_DIR", tmp_path)
    monkeypatch.setattr(download_module, "INPUT_META_DIR", tmp_path / ".metadata")

    task = {"name": "unique-recording"}
    attachment = {"title": "unique-recording.wav", "url": "https://example.com/unique-recording.wav"}

    with patch.object(download_module, "api_post") as mock_post:
        dest_path = _resolve_download_path(task, "task001", attachment)

    assert dest_path.name == "unique-recording.wav"
    mock_post.assert_not_called()


# ── Name mismatch check ─────────────────────────────────────────────────────


def test_check_name_mismatch_posts_comment_when_names_differ() -> None:
    task = {"name": "wrong-date-recording"}
    attachment = {"title": "correct-date-recording.wav", "url": ""}

    with patch.object(download_module, "api_post") as mock_post:
        _check_name_mismatch(task, "task001", attachment)

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "/task/task001/comment"
    body = call_args[0][1]
    assert "mismatch" in body["comment_text"].lower()
    assert "wrong-date-recording" in body["comment_text"]
    assert "correct-date-recording.wav" in body["comment_text"]


def test_check_name_mismatch_skips_when_names_match() -> None:
    task = {"name": "my-recording"}
    attachment = {"title": "my-recording.wav", "url": ""}

    with patch.object(download_module, "api_post") as mock_post:
        _check_name_mismatch(task, "task001", attachment)

    mock_post.assert_not_called()


def test_check_name_mismatch_case_insensitive() -> None:
    task = {"name": "My-Recording"}
    attachment = {"title": "my-recording.wav", "url": ""}

    with patch.object(download_module, "api_post") as mock_post:
        _check_name_mismatch(task, "task001", attachment)

    mock_post.assert_not_called()


def test_check_name_mismatch_uses_url_filename_when_no_title() -> None:
    task = {"name": "wrong-name"}
    attachment = {"title": "", "url": "https://example.com/correct-name.wav?dl=1"}

    with patch.object(download_module, "api_post") as mock_post:
        _check_name_mismatch(task, "task001", attachment)

    mock_post.assert_called_once()


def test_check_name_mismatch_skips_when_attachment_empty() -> None:
    task = {"name": "some-task"}
    attachment = {"title": "", "url": ""}

    with patch.object(download_module, "api_post") as mock_post:
        _check_name_mismatch(task, "task001", attachment)

    mock_post.assert_not_called()


def test_check_name_mismatch_handles_api_error_gracefully(capsys) -> None:
    task = {"name": "wrong-name"}
    attachment = {"title": "correct-name.wav", "url": ""}

    with patch.object(download_module, "api_post", side_effect=Exception("network error")):
        _check_name_mismatch(task, "task001", attachment)

    captured = capsys.readouterr()
    assert "Could not post mismatch comment" in captured.out
