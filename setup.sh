#!/bin/bash
# setup.sh — One-time setup for sacred-space pipeline
# Run once from inside the sacred-space/ folder:
#   chmod +x setup.sh && bash setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up sacred-space pipeline..."
echo ""

# Create working directories
mkdir -p "$SCRIPT_DIR/audio-input"
mkdir -p "$SCRIPT_DIR/audio-prepared"
mkdir -p "$SCRIPT_DIR/trim-artifacts"
mkdir -p "$SCRIPT_DIR/trim-experiments"
mkdir -p "$SCRIPT_DIR/audio-output"
echo "✓ Created audio-input/"
echo "✓ Created audio-prepared/"
echo "✓ Created trim-artifacts/"
echo "✓ Created trim-experiments/"
echo "✓ Created audio-output/"

# Create .env if missing
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  {
    echo "API_KEY=your_clickup_api_key_here"
    echo "VIEW_ID=your_clickup_view_id_here"
  } > "$SCRIPT_DIR/.env"
  echo "✓ Created .env (add your ClickUp API key and view ID)"
else
  echo "✓ .env already exists"
fi

# Check ffmpeg
if command -v ffmpeg &>/dev/null; then
  echo "✓ ffmpeg found"
else
  echo "⚠ ffmpeg not found — install with: brew install ffmpeg"
fi

if command -v ffprobe &>/dev/null; then
  echo "✓ ffprobe found"
else
  echo "⚠ ffprobe not found — install with: brew install ffmpeg"
fi

# Check deepFilter
if command -v deepFilter &>/dev/null; then
  echo "✓ DeepFilterNet found"
elif [ -f "/opt/anaconda3/envs/deepfilter/bin/deepFilter" ]; then
  echo "✓ DeepFilterNet found (conda)"
else
  echo "⚠ DeepFilterNet not found — run:"
  echo "    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu"
  echo "    pip install deepfilternet soundfile"
fi

echo ""
echo "Optional advanced trim backends:"
echo "  pip install webrtcvad-wheels faster-whisper"

echo ""
echo "════════════════════════════════════════"
echo "sacred-space/ is ready."
echo ""
echo "Pipeline:"
echo "  1.  python3 workflow1_download.py    # download WAVs from ClickUp"
echo "  2.  python3 prepare_raw_audio.py run # analyze/trim raw audio"
echo "  3.  bash process_audio.sh            # auto-runs prepare step, then de-noise + export MP3"
echo "  4.  python3 workflow2_upload.py      # upload MP3s back to ClickUp"
