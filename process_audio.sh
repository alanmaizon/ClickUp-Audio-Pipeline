#!/bin/bash
# process_audio.sh — Prepare raw audio → De-noise → Normalize → Silence → MP3
# Part of the sacred-space audio pipeline
#
# Input:   sacred-space/audio-input/{task_name_or_attachment_name}.{ext}
# Trimmed: sacred-space/audio-prepared/{task_name_or_attachment_name}.wav
# Output:  sacred-space/audio-output/{task_name_or_attachment_name}.mp3
#
# Steps:
#   1. prepare_raw_audio.py — intro/date removal + coarse silence trim
#   2. DeepFilterNet        — neural noise reduction
#   3. ffmpeg silenceremove — strip any residual leading silence
#   4. ffmpeg loudnorm      — -20 LUFS / peak -1dBTP
#   5. ffmpeg concat        — prepend exactly 4s silence lead-in
#   6. Export MP3           — 192k / 44.1kHz / stereo
#
# Usage:
#   bash process_audio.sh

set -euo pipefail

SKIP_PREPARE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-prepare)
      SKIP_PREPARE=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash process_audio.sh [--skip-prepare]"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1"
      echo "Usage: bash process_audio.sh [--skip-prepare]"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_INPUT_DIR="$SCRIPT_DIR/audio-input"
PREPARED_DIR="$SCRIPT_DIR/audio-prepared"
DENOISED_DIR="$SCRIPT_DIR/denoised-tmp"
OUTPUT_DIR="$SCRIPT_DIR/audio-output"
TRIM_CONFIG="$SCRIPT_DIR/config/trim_audio.toml"

if command -v deepFilter &>/dev/null; then
  DEEPFILTER="deepFilter"
elif [ -f "/opt/anaconda3/envs/deepfilter/bin/deepFilter" ]; then
  DEEPFILTER="/opt/anaconda3/envs/deepfilter/bin/deepFilter"
else
  echo "ERROR: deepFilter not found in PATH or /opt/anaconda3/envs/deepfilter/bin/"
  echo "Run: pip install deepfilternet soundfile"
  exit 1
fi

# ── Checks ────────────────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
  echo "ERROR: ffmpeg not found. Install: brew install ffmpeg"
  exit 1
fi

# ── Step 0: Prepare raw audio ─────────────────────────────────────────────────
if [ "$SKIP_PREPARE" -eq 0 ]; then
  echo "Step 0/3 — Prepare raw audio..."
  python3 "$SCRIPT_DIR/prepare_raw_audio.py" \
    --config "$TRIM_CONFIG" \
    run \
    --input-dir "$RAW_INPUT_DIR" \
    --output-dir "$PREPARED_DIR" \
    --artifact-dir "$SCRIPT_DIR/trim-artifacts" \
    --skip-existing
  echo ""
else
  echo "Step 0/3 — Skipping raw-audio prepare (already handled upstream)."
  echo ""
fi

# ── Count prepared WAVs ───────────────────────────────────────────────────────
shopt -s nullglob
wav_files=("$PREPARED_DIR"/*.wav)
count=${#wav_files[@]}

if [ "$count" -eq 0 ]; then
  echo "No prepared WAV files in $PREPARED_DIR — run workflow1_download.py first."
  exit 0
fi

mkdir -p "$PREPARED_DIR" "$DENOISED_DIR" "$OUTPUT_DIR"
echo "Processing $count file(s)..."
echo ""

# ── Step 1: De-noise ──────────────────────────────────────────────────────────
echo "Step 1/3 — DeepFilterNet de-noise..."
"$DEEPFILTER" "$PREPARED_DIR"/*.wav -o "$DENOISED_DIR/" 2>&1 || true
echo "Denoised files:"
ls -la "$DENOISED_DIR/" 2>/dev/null || echo "  (empty)"
echo ""

shopt -s nullglob
denoised_wavs=("$DENOISED_DIR"/*.wav)
if [ "${#denoised_wavs[@]}" -eq 0 ]; then
  echo "ERROR: DeepFilterNet produced no output — check installation (torch/torchaudio versions)."
  exit 1
fi

# ── Step 2: Normalize + Silence + Export MP3 ──────────────────────────────────
echo "Step 2/3 — Normalize + add silence + export MP3..."
processed=0
failed=0

for f in "$DENOISED_DIR"/*.wav; do
  [ -e "$f" ] || continue
  source_stem=$(basename "$f" .wav | sed 's/_DeepFilterNet3$//')
  if ! task_id=$(python3 "$SCRIPT_DIR/pipeline_naming.py" task-id-for-stem \
    --base-dir "$SCRIPT_DIR" \
    --stem "$source_stem"); then
    echo "    ✗ Failed: could not resolve task ID for ${source_stem}.wav"
    ((failed++)) || true
    continue
  fi
  output_name="${source_stem}.mp3"
  out="$OUTPUT_DIR/${output_name}"

  # Skip if already processed
  if [ -f "$out" ]; then
    echo "  ↷ ${output_name} already exists — skipping"
    continue
  fi

  echo "  → $output_name"
  if ffmpeg -y -i "$f" \
    -filter_complex "anullsrc=r=44100:cl=stereo:d=4[sil];[0:a]silenceremove=start_periods=1:start_silence=0:start_threshold=-35dB,loudnorm=I=-20:TP=-1:LRA=11,aresample=44100,asetpts=PTS-STARTPTS[aud];[sil][aud]concat=n=2:v=0:a=1[out]" \
    -map "[out]" -ac 2 -b:a 192k \
    "$out" -loglevel error 2>&1; then
    python3 "$SCRIPT_DIR/pipeline_naming.py" write-output-metadata \
      --base-dir "$SCRIPT_DIR" \
      --task-id "$task_id" \
      --output-filename "$output_name"
    echo "    ✓ ${output_name}"
    ((processed++)) || true
  else
    echo "    ✗ Failed: ${output_name}"
    ((failed++)) || true
  fi
done

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "Cleaning up temp files..."
rm -rf "$DENOISED_DIR"

echo ""
echo "════════════════════════════════════════"
echo "Processed: $processed  |  Failed: $failed"
echo "MP3s ready in: $OUTPUT_DIR"
echo ""
echo "Next → python3 workflow2_upload.py"
