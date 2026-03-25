# Raw Audio Trimming Stage

## Goal

Add a reproducible preprocessing stage between download and denoise/export so we can:

- remove boundary silence without touching interior pauses
- strip spoken opening dates when confidence is high
- preserve the real content onset conservatively
- compare methods with per-file evidence, manifests, and review shortlists

## Repo Insertion Point

Current flow:

1. `workflow1_download.py` downloads original attachments into `audio-input/`
2. `process_audio.sh` denoises and exports MP3s
3. `workflow2_upload.py` uploads final MP3s

New flow:

1. `run_pipeline.py`
2. `workflow1_download.py`
3. `prepare_raw_audio.py run`
4. `process_audio.sh --skip-prepare`
5. `workflow2_upload.py`

`process_audio.sh` now auto-runs the prepare step and consumes `audio-prepared/*.wav`, while keeping the original raw file untouched in `audio-input/`.

The download step writes `audio-input/.metadata/{stem}.json` with the original attachment title, task ID, and selection details so trim and upload can recover title/context hints without depending on opaque file names.

## Layered Decision System

### Layer A: ffmpeg silence baseline

- Uses `ffmpeg` `silencedetect` on the original file and again on a reversed copy
- Only trims boundaries
- Keeps configurable pad before the first speech onset and after the last speech region
- Produces raw tool evidence in the sidecar JSON

Why:

- zero extra Python dependency for the baseline
- fast and deterministic
- a useful lower-bound method for experiments

### Layer B: VAD-assisted boundary detection

- Preferred backend: `webrtcvad` / `webrtcvad-wheels`
- Decodes the file to 16 kHz mono PCM and groups speech-positive frames into spans
- Compares its opening/trailing boundary with the ffmpeg baseline
- In safe mode, disagreement lowers confidence and can trigger a conservative fallback

Why:

- better at ignoring short breaths/noise bursts that can fool pure silence detection
- still lightweight enough for batch experiments

### Layer C: ASR-guided intro removal

- Preferred backend: `faster-whisper`
- Only transcribes the first configured window, typically 15 to 30 seconds
- Classifies the opening speech into `date`, `title`, and `content`
- Uses regex and word timestamps to isolate only the `date` span
- Uses file-context hints and short-phrase heuristics to detect a title, but keeps it in the output
- Trims only the date span, with a tiny clamped post-date pad that cannot jump into the next spoken words

Why:

- silence detection alone cannot distinguish content speech from spoken metadata
- date-only removal is safer than a generic "trim the intro" heuristic because it preserves titles and the opening content
- the heuristics stay transparent and tunable instead of hiding the decision inside one opaque rule

## Safety Rules

- Raw source is never overwritten
- If no speech region is found confidently, keep the original window and flag manual review
- If VAD and ffmpeg disagree materially, safe mode prefers less destructive boundaries
- If intro removal would exceed the configured max automatic start trim, skip it and flag review
- If the trimmed result would be too short, fall back to the original window
- Intro classifier output is validated before manifest emission:
  only `date`, `title`, and `content` labels are allowed, spans must stay ordered, and only `date` may be removable
- File-level prepare failures are isolated into failure sidecars and manifest rows so one bad file does not abort the rest of the batch
- If a file fails during prepare, any stale prepared WAV for that file is removed to avoid accidental downstream reuse

## Outputs

Production run:

- `audio-prepared/{file_id}.wav`
- `trim-artifacts/files/{file_id}/{file_id}.trim.json`
- `trim-artifacts/events.jsonl`
- `trim-artifacts/manifest.jsonl`
- `trim-artifacts/manifest.csv`
- `trim-artifacts/summary.json`

Optional debug artifacts:

- waveform PNG
- opening transcript preview text
- opening `date | title | content` labels in the sidecar JSON
- raw backend spans and decisions inside the sidecar JSON

Experiment run:

- `trim-experiments/{run_id}/{method}/audio/*.wav`
- `trim-experiments/{run_id}/{method}/artifacts/...`
- `trim-experiments/{run_id}/manual-review-shortlist.json`
- `trim-experiments/{run_id}/report.md`

Failure rows are also written into the manifest and event log with `method_chosen = error`, and `summary.json` includes `files_failed`.

## Tradeoffs

- `ffmpeg` silence trimming is always available, but it cannot tell speech from a short breath or handling noise.
- `webrtcvad` adds a stronger speech-aware path, but it is optional and may not be installed in every environment.
- `faster-whisper` gives timestamped text for intro removal, but it is heavier; the code gates it behind an optional backend and degrades gracefully when it is missing.
- The heuristics deliberately prefer false negatives over clipped content.

## Optional Dependencies

Baseline only:

- `ffmpeg`
- `ffprobe`

Advanced mode:

```bash
pip install webrtcvad-wheels faster-whisper
```

After enabling advanced backends, force one refresh so older cached silence-only sidecars do not get reused:

```bash
python3 prepare_raw_audio.py run --method ffmpeg-vad-asr --force --debug
```

If `faster-whisper` has not downloaded a model yet, the first ASR run may need network access, or `asr.model_size` can point to a local model directory.

With `decision.fail_on_backend_fallback = true`, requesting `ffmpeg-vad` or `ffmpeg-vad-asr` will now stop the run if the required backend is unavailable instead of silently downgrading to ffmpeg-only trimming.

## Example Commands

Production mode:

```bash
python3 prepare_raw_audio.py run --debug --skip-existing
```

Production dry run:

```bash
python3 prepare_raw_audio.py run --dry-run --debug
```

Compare methods on the first 12 files:

```bash
python3 prepare_raw_audio.py experiment --sample-size 12
```

Tune a stricter silence baseline:

```bash
python3 prepare_raw_audio.py run \
  --method ffmpeg-silence \
  --min-silence-duration 0.5 \
  --silence-threshold-db -38 \
  --start-pad 0.10 \
  --end-pad 0.15
```
