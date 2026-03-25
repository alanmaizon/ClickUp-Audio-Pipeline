# ClickUp Audio Pipeline

Small local pipeline for:

1. downloading raw recordings from ClickUp
2. trimming boundary silence and optional spoken opening dates while preserving titles
3. denoising and exporting polished MP3s
4. uploading the finished MP3s back to ClickUp

## Workflow

1. Run setup once:

```bash
bash setup.sh
```

2. Run the end-to-end pipeline:

```bash
python3 run_pipeline.py
```

That single command runs:

1. `workflow1_download.py`
2. `prepare_raw_audio.py run`
3. `process_audio.sh --skip-prepare`
4. `workflow2_upload.py`

Useful end-to-end variations:

```bash
python3 run_pipeline.py --stop-after prepare
python3 run_pipeline.py --skip-upload
python3 run_pipeline.py --trim-method ffmpeg-vad-asr --trim-debug --trim-force
python3 run_pipeline.py --clean
```

Use `--stop-after prepare` for a safe first real-data pass if you want to inspect `trim-artifacts/manifest.csv` and a few sidecars before denoise/export/upload.

Use `--clean` to automatically remove audio working files after a successful upload, keeping the workspace ready for the next session. Metadata sidecars are preserved.

## Individual Stages

If you want to run or debug the stages manually:

```bash
python3 workflow1_download.py
python3 prepare_raw_audio.py run --debug --skip-existing
bash process_audio.sh
python3 workflow2_upload.py
```

`process_audio.sh` auto-runs the raw-audio prepare step, so step 3 is optional if you just want the end-to-end flow.

## Key Directories

- `audio-input/`: original downloaded source audio using the task/attachment stem
- `audio-input/.metadata/`: download-time metadata with ClickUp `task_id` and attachment details
- `audio-prepared/`: trimmed WAVs using the same human-readable stem
- `audio-prepared/.metadata/`: trimmed-audio metadata used to recover `task_id` downstream
- `audio-output/`: final MP3 exports using the same human-readable stem
- `trim-artifacts/`: sidecars, manifests, logs, debug artifacts
- `trim-experiments/`: experiment runs comparing trim methods

## Trimming Stage

`prepare_raw_audio.py` adds a layered preprocessing stage:

- ffmpeg silence baseline
- optional WebRTC VAD boundary detection
- optional Faster-Whisper intro classification for `date | title | content`

When ASR is enabled, the opening transcript is classified so that:

- `date` is removable
- `title` is detected but preserved
- `content` is preserved

Operational safeguards:

- malformed `classified_segments` are rejected before they can affect trim metadata
- a single-file prepare failure is written to a failure sidecar and manifest row instead of aborting the whole batch
- if a file fails during prepare, any stale prepared WAV for that file is removed so downstream processing does not reuse it accidentally

Useful commands:

```bash
python3 prepare_raw_audio.py run --dry-run --debug
python3 prepare_raw_audio.py experiment --sample-size 12
```

Configuration lives in `config/trim_audio.toml`.

The design note is in `docs/raw-audio-trimming.md`.

Download and naming behavior:

- filenames are derived from the **audio attachment filename**, not the volunteer-entered task name, to prevent date/name errors from propagating through the pipeline
- if the task name and attachment filename don't match, a note is posted on the ClickUp task activity panel so the team can review
- if two tasks share the same attachment filename, the pipeline appends the task ID to disambiguate and posts a note on the affected task
- the same stem is preserved through `audio-prepared/` and `audio-output/`
- ClickUp `task_id` lives in metadata sidecars, so upload matching stays reliable even though the visible files are human-readable
- the downloader chooses audio attachments deterministically, prefers non-`copy` variants, and preserves the real extension from the source attachment

## Audio Processing

`process_audio.sh` runs after trimming and applies:

1. DeepFilterNet neural noise reduction
2. Residual leading silence removal (catches anything the trim step missed)
3. Loudness normalization to -20 LUFS / -1 dBTP
4. Exactly 4 seconds of clean silence prepended for a consistent start
5. Export as 192kbps stereo MP3 at 44.1kHz

Every output has the same format and lead-in regardless of how the raw recording was captured.

## Session Workflow

Each batch is a self-contained session:

1. `python3 run_pipeline.py` — downloads, trims, processes, uploads
2. `python3 run_pipeline.py --clean` — same, but clears audio folders after upload

The `--clean` flag removes files from `audio-input/`, `audio-prepared/`, `audio-output/`, and `trim-artifacts/` but preserves `.metadata/` sidecars. To review output before uploading, use `--stop-after process` first.

## Optional Dependencies

Baseline trimming only needs `ffmpeg` and `ffprobe`.

For advanced VAD and ASR trimming:

```bash
pip install webrtcvad-wheels faster-whisper
```

After installing advanced backends, do one forced re-run so older silence-only sidecars are refreshed:

```bash
python3 prepare_raw_audio.py run --method ffmpeg-vad-asr --debug --force
```

`faster-whisper` also needs a Whisper model available locally. On the first run it may download the configured model, or you can point `config/trim_audio.toml` `asr.model_size` at a local model path.

When `ffmpeg-vad` or `ffmpeg-vad-asr` is requested, the pipeline now fails fast if the required backend is not actually ready. It will not silently fall back to plain ffmpeg trimming.

## Git Hygiene

Raw recordings, prepared audio, exports, and experiment artifacts are intentionally ignored by git in `.gitignore`.
