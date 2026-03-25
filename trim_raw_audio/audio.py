from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .models import AudioInfo


class AudioCommandError(RuntimeError):
    pass


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    completed = subprocess.run(
        list(args),
        check=False,
        capture_output=capture_output,
        text=text,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr if isinstance(completed.stderr, str) else ""
        raise AudioCommandError(f"Command failed ({completed.returncode}): {' '.join(args)}\n{stderr}")
    return completed


def probe_audio(path: Path) -> AudioInfo:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name:format=duration,format_name",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or [{}]
    stream = streams[0]
    fmt = payload.get("format") or {}
    return AudioInfo(
        path=path,
        duration_sec=float(fmt.get("duration") or 0.0),
        sample_rate_hz=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        codec_name=stream.get("codec_name") or "",
        format_name=fmt.get("format_name") or "",
    )


def run_silencedetect(
    path: Path,
    *,
    noise_threshold_db: float,
    min_silence_duration_sec: float,
    reverse: bool = False,
) -> str:
    filters = []
    if reverse:
        filters.append("areverse")
    filters.append(
        f"silencedetect=n={noise_threshold_db}dB:d={min_silence_duration_sec}"
    )
    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            ",".join(filters),
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    return completed.stderr or ""


def extract_pcm_mono_16k(path: Path) -> bytes:
    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        text=False,
    )
    return completed.stdout or b""


def trim_audio_to_wav(
    input_path: Path,
    output_path: Path,
    *,
    start_offset_sec: float,
    end_offset_sec: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = max(0.0, end_offset_sec - start_offset_sec)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-ss",
            f"{start_offset_sec:.3f}",
            "-t",
            f"{duration_sec:.3f}",
            "-i",
            str(input_path),
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def render_excerpt_for_asr(
    input_path: Path,
    output_path: Path,
    *,
    analysis_window_sec: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-t",
            f"{analysis_window_sec:.3f}",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def render_waveform_png(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-lavfi",
            "showwavespic=s=1600x240:colors=0x2874A6",
            "-frames:v",
            "1",
            str(output_path),
        ]
    )
