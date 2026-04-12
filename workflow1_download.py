#!/usr/bin/env python3
"""
WORKFLOW 1 — Download RECORDED tasks
======================================
Fetches all tasks with status RECORDED, downloads the selected audio attachment
to sacred-space/audio-input/{task_name_or_attachment_name}.{ext}, then sets
status to 'in progress'.

Human-readable filenames are preserved across the pipeline. Task IDs stay in
sidecar metadata for ClickUp-safe upload matching.

Usage:
    python3 workflow1_download.py

Requires .env in sacred-space/ with:
    API_KEY=pk_xxxxx
    VIEW_ID=x-xxxxxxxx-x
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib import error, parse, request

from pipeline_naming import load_json, preferred_source_stem, sanitize_filename_stem

# ── Load .env ─────────────────────────────────────────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip()

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY     = os.environ.get("API_KEY") or os.environ.get("CLICKUP_API_TOKEN")
VIEW_ID     = os.environ.get("VIEW_ID")
BASE_URL    = "https://api.clickup.com/api/v2"
BASE_DIR    = Path(__file__).parent
INPUT_DIR   = BASE_DIR / "audio-input"
INPUT_META_DIR = INPUT_DIR / ".metadata"
STATUS_FROM = "RECORDED"
STATUS_TO   = "in progress"
AUDIO_EXTS  = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3",
               ".ogg", ".opus", ".wav", ".wma"}
RETRYABLE   = {429, 500, 502, 503, 504}
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
MIN_DOWNLOAD_BYTES = 1024               # 1 KB — anything smaller is likely corrupt
_COPY_MARKER_RE = re.compile(r"(?:^|[\s_.-])(copy|duplicate)(?:$|[\s_.-])|\(\d+\)$", re.IGNORECASE)
# ──────────────────────────────────────────────────────────────────────────────


def headers(content_type: str = "application/json") -> dict:
    if not API_KEY:
        sys.exit("ERROR: API_KEY not found in .env")
    if not VIEW_ID:
        sys.exit("ERROR: VIEW_ID not found in .env")
    return {"Authorization": API_KEY, "Content-Type": content_type,
            "User-Agent": "sacred-space/1.0"}


def api_get(path: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + parse.urlencode(params, doseq=True)
    req = request.Request(url, headers=headers())
    for attempt in range(5):
        try:
            with request.urlopen(req) as r:
                return json.loads(r.read())
        except error.HTTPError as e:
            if e.code not in RETRYABLE or attempt == 4:
                raise
            delay = int(e.headers.get("Retry-After") or 2 ** attempt)
            print(f"  HTTP {e.code} — retrying in {delay}s...")
            time.sleep(delay)


def api_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = request.Request(f"{BASE_URL}{path}", data=data,
                          headers=headers(), method="POST")
    for attempt in range(5):
        try:
            with request.urlopen(req) as r:
                return json.loads(r.read())
        except error.HTTPError as e:
            if e.code not in RETRYABLE or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    return {}


def api_put(path: str, body: dict) -> None:
    data = json.dumps(body).encode()
    req = request.Request(f"{BASE_URL}{path}", data=data,
                          headers=headers(), method="PUT")
    for attempt in range(5):
        try:
            with request.urlopen(req) as r:
                r.read()
                return
        except error.HTTPError as e:
            if e.code not in RETRYABLE or attempt == 4:
                raise
            time.sleep(2 ** attempt)


MAX_PAGES = 50


def fetch_recorded_tasks() -> list[dict]:
    tasks, seen, page = [], set(), 0
    while page < MAX_PAGES:
        data = api_get(f"/view/{VIEW_ID}/task", {
            "page": page, "include_closed": "false",
            "statuses[]": [STATUS_FROM],
        })
        batch = data.get("tasks", [])
        for t in batch:
            tid = t.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                status = (t.get("status") or {}).get("status", "")
                if status.upper() == STATUS_FROM.upper():
                    tasks.append(t)
        if not batch or data.get("last_page") is True or len(batch) < 100:
            break
        page += 1
    return tasks


def _check_name_mismatch(task: dict, task_id: str, attachment: dict) -> None:
    task_name = sanitize_filename_stem(task.get("name", ""))
    att_title = attachment.get("title") or _attachment_url_name(attachment.get("url", ""))
    att_stem = sanitize_filename_stem(att_title)
    if not task_name or not att_stem:
        return
    if task_name.lower() == att_stem.lower():
        return
    note = (
        f"⚠ Name mismatch detected by pipeline:\n"
        f"  Task name: {task.get('name', '')}\n"
        f"  Audio file: {att_title}\n"
        f"Pipeline used the audio filename. Please verify the task name is correct."
    )
    try:
        api_post(f"/task/{task_id}/comment", {"comment_text": note})
        print(f"    📝 Name mismatch noted in task activity")
    except Exception as e:
        print(f"    ⚠ Could not post mismatch comment: {e}")


def _candidate_audio_extension(title: str, url: str) -> str | None:
    url_path = url.split("?", 1)[0]
    url_ext = Path(url_path).suffix.lower()
    if url_ext in AUDIO_EXTS:
        return url_ext
    title_ext = Path(title).suffix.lower()
    if title_ext in AUDIO_EXTS:
        return title_ext
    return None


def _copy_like_score(value: str) -> int:
    normalized = value.strip().lower()
    if not normalized:
        return 0
    return 1 if _COPY_MARKER_RE.search(normalized) else 0


def _attachment_timestamp(att: dict) -> int:
    for key in ("date", "date_created", "date_uploaded", "created_at"):
        value = att.get(key)
        if value is None:
            continue
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return 0


def _attachment_url_name(url: str) -> str:
    return Path(url.split("?", 1)[0]).name


def _attachment_summary(att: dict) -> dict:
    title = att.get("title", "") or ""
    url = att.get("url", "") or ""
    ext = _candidate_audio_extension(title, url)
    url_name = _attachment_url_name(url)
    copy_like = max(_copy_like_score(title), _copy_like_score(url_name))
    return {
        "id": att.get("id"),
        "title": title,
        "url": url,
        "url_filename": url_name,
        "audio_ext": ext or "",
        "timestamp": _attachment_timestamp(att),
        "copy_like": bool(copy_like),
    }


def _attachment_sort_key(att: dict) -> tuple[int, int, int, str]:
    summary = _attachment_summary(att)
    has_title_ext = 1 if Path(summary["title"]).suffix.lower() in AUDIO_EXTS else 0
    return (
        0 if not summary["copy_like"] else 1,
        -summary["timestamp"],
        -has_title_ext,
        summary["title"].lower() or summary["url_filename"].lower(),
    )


def get_audio_attachments(task_id: str) -> list[dict]:
    try:
        task = api_get(f"/task/{task_id}")
        audio_attachments: list[dict] = []
        for att in (task.get("attachments") or []):
            title = att.get("title", "") or ""
            url   = att.get("url", "") or ""
            if _candidate_audio_extension(title, url):
                audio_attachments.append(att)
        return audio_attachments
    except Exception as e:
        print(f"    ⚠ Could not fetch task: {e}")
    return []


def select_audio_attachment(attachments: list[dict]) -> dict | None:
    if not attachments:
        return None
    return sorted(attachments, key=_attachment_sort_key)[0]


def _metadata_path(stem: str) -> Path:
    return INPUT_META_DIR / f"{stem}.json"


def _write_source_metadata(*, task: dict, task_id: str, selected_attachment: dict, attachments: list[dict], dest_path: Path) -> None:
    INPUT_META_DIR.mkdir(parents=True, exist_ok=True)
    selected_summary = _attachment_summary(selected_attachment)
    candidate_summaries = [_attachment_summary(att) for att in attachments]
    payload = {
        "task_id": task_id,
        "task_name": task.get("name", ""),
        "download_filename": dest_path.name,
        "download_stem": dest_path.stem,
        "selected_attachment": selected_summary,
        "candidate_attachments": candidate_summaries,
    }
    _metadata_path(dest_path.stem).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metadata_records_for_task(task_id: str) -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    if not INPUT_META_DIR.exists():
        return records
    for path in sorted(INPUT_META_DIR.glob("*.json")):
        payload = load_json(path)
        if payload.get("task_id") == task_id:
            records.append((path, payload))
    return records


def _existing_audio_download(task_id: str) -> Path | None:
    for _meta_path, payload in _metadata_records_for_task(task_id):
        download_filename = payload.get("download_filename")
        if not download_filename:
            continue
        path = INPUT_DIR / download_filename
        if path.exists() and path.is_file():
            return path

    # Backward-compatible fallback for older task-id-based downloads.
    for path in sorted(INPUT_DIR.iterdir()):
        if path.is_file() and path.stem == task_id and path.suffix.lower() in AUDIO_EXTS:
            return path
    return None


def _preferred_download_stem(task: dict, task_id: str, attachment: dict) -> str:
    summary = _attachment_summary(attachment)
    return preferred_source_stem(task_id, task.get("name", ""), summary)


def _resolve_download_path(task: dict, task_id: str, attachment: dict) -> Path:
    raw_title = attachment.get("title") or _attachment_url_name(attachment.get("url", ""))
    ext = _candidate_audio_extension(raw_title, attachment.get("url", "")) or ".wav"
    preferred_stem = _preferred_download_stem(task, task_id, attachment)

    candidate_stems = [preferred_stem, f"{preferred_stem}-{task_id}"]
    for stem in candidate_stems:
        dest_path = INPUT_DIR / f"{stem}{ext}"
        owner_task_id = load_json(_metadata_path(stem)).get("task_id")
        if owner_task_id in {"", None, task_id}:
            if stem != preferred_stem:
                _note_duplicate_name(task_id, preferred_stem, dest_path.name)
            return dest_path
        if not dest_path.exists():
            if stem != preferred_stem:
                _note_duplicate_name(task_id, preferred_stem, dest_path.name)
            return dest_path

    index = 2
    while True:
        stem = f"{preferred_stem}-{task_id}-{index}"
        dest_path = INPUT_DIR / f"{stem}{ext}"
        owner_task_id = load_json(_metadata_path(stem)).get("task_id")
        if owner_task_id in {"", None, task_id} or not dest_path.exists():
            _note_duplicate_name(task_id, preferred_stem, dest_path.name)
            return dest_path
        index += 1


def _note_duplicate_name(task_id: str, preferred_stem: str, actual_filename: str) -> None:
    note = (
        f"⚠ Duplicate filename detected by pipeline:\n"
        f"  Another task already uses: {preferred_stem}\n"
        f"  This task was saved as: {actual_filename}\n"
        f"Please check that this recording has a unique name."
    )
    try:
        api_post(f"/task/{task_id}/comment", {"comment_text": note})
        print(f"    📝 Duplicate name noted in task activity")
    except Exception as e:
        print(f"    ⚠ Could not post duplicate comment: {e}")


def download(url: str, dest: Path) -> None:
    req = request.Request(url, headers={"User-Agent": "sacred-space/1.0"})
    with request.urlopen(req) as r:
        chunks = []
        total = 0
        while True:
            chunk = r.read(8192)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise RuntimeError(f"Download exceeds {MAX_DOWNLOAD_BYTES // (1024*1024)} MB limit")
            chunks.append(chunk)
        if total < MIN_DOWNLOAD_BYTES:
            raise RuntimeError(f"Download too small ({total} bytes) — likely corrupt or empty")
        dest.write_bytes(b"".join(chunks))


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_META_DIR.mkdir(parents=True, exist_ok=True)
    print(f"sacred-space → audio-input/\n")

    total_processed = total_skipped = total_errors = 0

    while True:
        print(f"Checking for '{STATUS_FROM}' tasks...")
        tasks = fetch_recorded_tasks()
        if not tasks:
            print(f"No '{STATUS_FROM}' tasks remaining. All done.\n")
            break

        print(f"Found {len(tasks)} task(s).\n")

        for task in tasks:
            name    = task.get("name", task["id"])
            task_id = task["id"]
            print(f"  [{task_id}] {name}")

            # Skip if already downloaded
            existing_audio = _existing_audio_download(task_id)
            if existing_audio:
                print(f"    ↷ Already downloaded ({existing_audio.name}) — updating status only")
                try:
                    if not _metadata_records_for_task(task_id):
                        attachments = get_audio_attachments(task_id)
                        selected = select_audio_attachment(attachments)
                        if selected:
                            _write_source_metadata(
                                task=task,
                                task_id=task_id,
                                selected_attachment=selected,
                                attachments=attachments,
                                dest_path=existing_audio,
                            )
                    api_put(f"/task/{task_id}", {"status": STATUS_TO})
                    print(f"    ✓ Status updated\n")
                    total_processed += 1
                except Exception as e:
                    print(f"    ✗ Status update failed: {e}\n")
                    total_errors += 1
                time.sleep(0.3)
                continue

            try:
                attachments = get_audio_attachments(task_id)
                att = select_audio_attachment(attachments)
                if not att:
                    print(f"    ⚠ No audio attachment — skipping\n")
                    total_skipped += 1
                    continue

                dest_path = _resolve_download_path(task, task_id, att)

                print(f"    ↓ {dest_path.name}")
                if len(attachments) > 1:
                    selected_label = att.get("title") or _attachment_url_name(att.get("url", ""))
                    print(f"    ↳ Selected attachment: {selected_label}")
                download(att["url"], dest_path)
                _write_source_metadata(
                    task=task,
                    task_id=task_id,
                    selected_attachment=att,
                    attachments=attachments,
                    dest_path=dest_path,
                )
                _check_name_mismatch(task, task_id, att)

                print(f"    ✎ RECORDED → in progress")
                api_put(f"/task/{task_id}", {"status": STATUS_TO})

                print(f"    ✓ Done\n")
                total_processed += 1
                time.sleep(0.4)

            except Exception as exc:
                print(f"    ✗ Error: {exc}\n")
                total_errors += 1

    print("=" * 50)
    print(f"Processed: {total_processed}  Skipped: {total_skipped}  Errors: {total_errors}")
    print(f"Files in: {INPUT_DIR}")
    print(f"\nNext → python3 prepare_raw_audio.py run")
    print("        or just run: bash process_audio.sh  # auto-runs raw-audio prep")


if __name__ == "__main__":
    main()
