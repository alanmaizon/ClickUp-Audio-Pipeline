#!/usr/bin/env python3
"""
WORKFLOW 2 — Upload edited MP3s
==================================
Fetches all tasks with status 'in progress', looks for a matching MP3
in sacred-space/audio-output/{task_name_or_attachment_name}.mp3, uploads it, sets
status to 'edited'.

Human-readable filenames are resolved back to ClickUp task IDs through
audio-output/.metadata/{task_id}.json.

Usage:
    python3 workflow2_upload.py

Requires .env in sacred-space/ with:
    API_KEY=pk_xxxxx
    VIEW_ID=x-xxxxxxxx-x
"""

from __future__ import annotations

import json
import os
import sys
import time
import mimetypes
import uuid
from pathlib import Path
from urllib import error, parse, request

from pipeline_naming import resolve_output_for_upload

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
OUTPUT_DIR  = BASE_DIR / "audio-output"
STATUS_FROM = "in progress"
STATUS_TO   = "edited"
RETRYABLE   = {429, 500, 502, 503, 504}
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


def upload_attachment(task_id: str, file_path: Path) -> None:
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(file_path.name)[0] or "audio/mpeg"
    file_data = file_path.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="attachment"; filename="{file_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = request.Request(
        f"{BASE_URL}/task/{task_id}/attachment",
        data=body,
        headers={"Authorization": API_KEY,
                 "Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "sacred-space/1.0"},
        method="POST",
    )
    for attempt in range(5):
        try:
            with request.urlopen(req) as r:
                r.read()
                return
        except error.HTTPError as e:
            if e.code not in RETRYABLE or attempt == 4:
                raise
            time.sleep(2 ** attempt)


def fetch_inprogress_tasks() -> list[dict]:
    tasks, seen, page = [], set(), 0
    while True:
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
                if status.lower() == STATUS_FROM.lower():
                    tasks.append(t)
        if not batch or data.get("last_page") is True or len(batch) < 100:
            break
        page += 1
    return tasks


def main():
    if not OUTPUT_DIR.exists():
        print(f"audio-output/ folder not found in sacred-space/")
        print("Run process_audio.sh first.")
        sys.exit(1)

    available_mp3s = list(OUTPUT_DIR.glob("*.mp3"))
    if not available_mp3s:
        print(f"No MP3s in {OUTPUT_DIR} — run process_audio.sh first.")
        sys.exit(1)

    print(f"Found {len(available_mp3s)} MP3(s) in audio-output/\n")

    total_uploaded = total_skipped = total_errors = 0

    while True:
        print(f"Checking for '{STATUS_FROM}' tasks...")
        tasks = fetch_inprogress_tasks()
        if not tasks:
            print(f"No '{STATUS_FROM}' tasks remaining. All done.\n")
            break

        print(f"Found {len(tasks)} task(s).\n")
        made_progress = False

        for task in tasks:
            name    = task.get("name", task["id"])
            task_id = task["id"]
            print(f"  [{task_id}] {name}")

            mp3 = resolve_output_for_upload(BASE_DIR, task_id)
            if not mp3:
                print(f"    ⚠ No MP3 found for {task_id} in audio-output/ — skipping\n")
                total_skipped += 1
                continue

            try:
                print(f"    ↑ Uploading: {mp3.name}")
                upload_attachment(task_id, mp3)

                print(f"    ✎ in progress → edited")
                api_put(f"/task/{task_id}", {"status": STATUS_TO})

                print(f"    ✓ Done\n")
                total_uploaded += 1
                made_progress = True
                time.sleep(0.4)

            except Exception as exc:
                print(f"    ✗ Error: {exc}\n")
                total_errors += 1

        if not made_progress:
            print("\nNo further matches. Add remaining MP3s to audio-output/ and re-run.")
            break

    print("=" * 50)
    print(f"Uploaded: {total_uploaded}  Skipped: {total_skipped}  Errors: {total_errors}")


if __name__ == "__main__":
    main()
