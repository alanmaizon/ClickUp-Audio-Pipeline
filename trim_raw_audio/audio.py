from __future__ import annotations

import math
import json
import subprocess
from array import array
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


def extract_pcm_mono_excerpt(
    path: Path,
    *,
    start_offset_sec: float,
    duration_sec: float,
    sample_rate_hz: int = 16000,
) -> tuple[array, int]:
    if duration_sec <= 0:
        return array("h"), sample_rate_hz

    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-vn",
            "-af",
            f"atrim=start={start_offset_sec:.6f}:duration={duration_sec:.6f},asetpts=PTS-STARTPTS",
            "-ac",
            "1",
            "-ar",
            str(sample_rate_hz),
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        text=False,
    )
    samples = array("h")
    payload = completed.stdout or b""
    if payload:
        samples.frombytes(payload)
    return samples, sample_rate_hz


def _rms_dbfs(samples: Sequence[int]) -> float:
    if not samples:
        return -120.0
    energy = sum(sample * sample for sample in samples) / len(samples)
    if energy <= 1e-12:
        return -120.0
    rms = math.sqrt(energy) / 32768.0
    return 20.0 * math.log10(max(rms, 1e-9))


def _first_zero_crossing(samples: Sequence[int], start_index: int, end_index: int) -> int | None:
    if not samples:
        return None
    start_index = max(1, start_index)
    end_index = min(len(samples), end_index)
    for index in range(start_index, end_index):
        previous = samples[index - 1]
        current = samples[index]
        if previous == 0 or current == 0:
            return index
        if (previous < 0 <= current) or (previous > 0 >= current):
            return index
    return None


def _last_zero_crossing(samples: Sequence[int], start_index: int, end_index: int) -> int | None:
    if not samples:
        return None
    start_index = max(1, start_index)
    end_index = min(len(samples), end_index)
    for index in range(end_index - 1, start_index - 1, -1):
        previous = samples[index - 1]
        current = samples[index]
        if previous == 0 or current == 0:
            return index
        if (previous < 0 <= current) or (previous > 0 >= current):
            return index
    return None


def find_opening_boundary_snap(
    input_path: Path,
    *,
    search_start_sec: float,
    search_end_sec: float,
    frame_ms: int,
    threshold_db: float,
    direction: str = "forward",
) -> dict[str, Any]:
    search_window_sec = max(0.0, search_end_sec - search_start_sec)
    if direction not in {"forward", "backward"}:
        raise ValueError(f"Unsupported snap direction: {direction}")
    result: dict[str, Any] = {
        "searched": False,
        "found": False,
        "reason": "",
        "search_start_sec": round(search_start_sec, 3),
        "search_end_sec": round(search_end_sec, 3),
        "frame_ms": frame_ms,
        "threshold_db": threshold_db,
        "direction": direction,
        "snapped_trim_start_sec": round(search_start_sec, 3),
        "frame_db": None,
        "zero_crossing_used": False,
    }
    if search_window_sec <= 0:
        result["reason"] = "boundary snap window was empty"
        return result

    samples, sample_rate_hz = extract_pcm_mono_excerpt(
        input_path,
        start_offset_sec=search_start_sec,
        duration_sec=search_window_sec,
    )
    result["searched"] = True
    result["sample_rate_hz"] = sample_rate_hz
    result["sample_count"] = len(samples)
    if not samples:
        result["reason"] = "boundary snap analysis produced no audio samples"
        return result

    frame_size = max(1, int(sample_rate_hz * (frame_ms / 1000.0)))
    best_db = float("inf")
    best_frame_start = 0
    frame_starts = list(range(0, len(samples), frame_size))
    search_order = frame_starts if direction == "forward" else list(reversed(frame_starts))

    for frame_start in search_order:
        frame_end = min(len(samples), frame_start + frame_size)
        frame = samples[frame_start:frame_end]
        frame_db = _rms_dbfs(frame)
        if frame_db < best_db:
            best_db = frame_db
            best_frame_start = frame_start
        if frame_db > threshold_db:
            continue

        if direction == "forward":
            zero_crossing = _first_zero_crossing(samples, frame_start, frame_end)
            snap_index = zero_crossing if zero_crossing is not None else frame_start
        else:
            zero_crossing = _last_zero_crossing(samples, frame_start, frame_end)
            snap_index = zero_crossing if zero_crossing is not None else max(frame_start, frame_end - 1)
        result.update(
            {
                "found": True,
                "reason": "zero crossing near low-energy dip" if zero_crossing is not None else "low-energy frame",
                "frame_db": round(frame_db, 3),
                "zero_crossing_used": zero_crossing is not None,
                "snapped_trim_start_sec": round(search_start_sec + (snap_index / sample_rate_hz), 3),
            }
        )
        return result

    result["frame_db"] = round(best_db, 3)
    result["reason"] = (
        "no low-energy snap point found inside the allowed overshoot window"
        if direction == "forward"
        else "no low-energy snap point found inside the allowed preroll window"
    )
    result["best_frame_start_sec"] = round(search_start_sec + (best_frame_start / sample_rate_hz), 3)
    return result


def trim_audio_to_wav(
    input_path: Path,
    output_path: Path,
    *,
    start_offset_sec: float,
    end_offset_sec: float,
    leading_fade_in_ms: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = max(0.0, end_offset_sec - start_offset_sec)
    filters = [
        f"atrim=start={start_offset_sec:.6f}:end={end_offset_sec:.6f}",
        "asetpts=PTS-STARTPTS",
    ]
    if leading_fade_in_ms > 0 and duration_sec > 0:
        fade_duration_sec = min(duration_sec, leading_fade_in_ms / 1000.0)
        filters.append(f"afade=t=in:st=0:d={fade_duration_sec:.6f}")
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            ",".join(filters),
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
