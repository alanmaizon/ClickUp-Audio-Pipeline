#!/usr/bin/env python3
"""Denoise WAV files using noisereduce (drop-in replacement for deepFilter CLI).

Usage:
    python3 denoise_audio.py INPUT_DIR OUTPUT_DIR

Reads all .wav files from INPUT_DIR, applies stationary noise reduction,
and writes denoised .wav files to OUTPUT_DIR with the original filename
(no suffix appended).
"""
from __future__ import annotations

import sys
from pathlib import Path

import noisereduce as nr
import soundfile as sf


def denoise_file(src: Path, dst: Path) -> None:
    audio, sr = sf.read(src)
    reduced = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)
    sf.write(dst, reduced, sr)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} INPUT_DIR OUTPUT_DIR", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files in {input_dir}")
        sys.exit(1)

    for src in wav_files:
        dst = output_dir / src.name
        print(f"  Denoising: {src.name}")
        denoise_file(src, dst)
        print(f"    done")

    print(f"\nDenoised {len(wav_files)} file(s) → {output_dir}")


if __name__ == "__main__":
    main()
