from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .audio import probe_audio, render_waveform_png, trim_audio_to_wav
from .backends import FasterWhisperAsrBackend, FfmpegSilenceBackend, WebRtcVadBackend
from .config import TrimConfig
from .intro import detect_intro
from .models import BoundaryEvidence, DecisionRecord, IntroDetectionResult, TranscriptResult


SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
METHODS = ("ffmpeg-silence", "ffmpeg-vad", "ffmpeg-vad-asr")


class BackendNotReadyError(RuntimeError):
    """Raised when a requested trim method cannot run with the active backends."""


class IntroDetectionValidationError(ValueError):
    """Raised when intro classification output is structurally invalid."""


_ALLOWED_INTRO_LABELS = {"date", "title", "content"}
_TRAILING_COPY_RE = re.compile(r"(?:[\s_-]*copy|\(\d+\))$", re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_NEXT_BOUNDARY_INTRUSION_SEC = 0.18


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_cached_sidecar(sidecar_path: Path) -> dict[str, Any] | None:
    if not sidecar_path.exists():
        return None
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _transcript_snippet(transcript: TranscriptResult | None) -> str:
    if not transcript or not transcript.text:
        return ""
    return transcript.text[:200]


def _classified_span_bounds(intro: IntroDetectionResult, label: str) -> tuple[float | None, float | None]:
    spans = [segment for segment in intro.classified_segments if segment.label == label]
    if not spans:
        return None, None
    return spans[0].start_sec, spans[-1].end_sec


def _opening_labels(intro: IntroDetectionResult) -> str:
    labels: list[str] = []
    for segment in intro.classified_segments:
        if not labels or labels[-1] != segment.label:
            labels.append(segment.label)
    return "|".join(labels)


def _validate_intro_detection(intro: IntroDetectionResult) -> None:
    previous_end: float | None = None
    date_segments = [segment for segment in intro.classified_segments if segment.label == "date"]

    for index, segment in enumerate(intro.classified_segments):
        if segment.label not in _ALLOWED_INTRO_LABELS:
            raise IntroDetectionValidationError(
                f"classified_segments[{index}] has unsupported label '{segment.label}'"
            )
        if segment.start_sec < 0:
            raise IntroDetectionValidationError(
                f"classified_segments[{index}] starts before 0 seconds"
            )
        if segment.end_sec < segment.start_sec:
            raise IntroDetectionValidationError(
                f"classified_segments[{index}] ends before it starts"
            )
        if previous_end is not None and segment.start_sec < previous_end - 0.01:
            raise IntroDetectionValidationError(
                f"classified_segments[{index}] overlaps or goes backward in time"
            )
        if segment.removable and segment.label != "date":
            raise IntroDetectionValidationError(
                f"classified_segments[{index}] is removable but labeled '{segment.label}'"
            )
        previous_end = segment.end_sec

    if intro.detected:
        if intro.trim_offset_sec is None:
            raise IntroDetectionValidationError("intro.detected is true but trim_offset_sec is missing")
        if not date_segments:
            raise IntroDetectionValidationError(
                "intro.detected is true but no removable date segment was provided"
            )
        expected_trim = max(segment.end_sec for segment in date_segments)
        if abs(expected_trim - intro.trim_offset_sec) > 0.05:
            raise IntroDetectionValidationError(
                "intro.trim_offset_sec does not match the end of the classified date span"
            )


def _safe_probe_audio(input_path: Path) -> dict[str, Any]:
    try:
        audio_info = probe_audio(input_path)
    except Exception:
        return {
            "duration_sec": None,
            "sample_rate_hz": None,
            "channels": None,
            "codec_name": "",
            "format_name": "",
        }

    return {
        "duration_sec": audio_info.duration_sec,
        "sample_rate_hz": audio_info.sample_rate_hz,
        "channels": audio_info.channels,
        "codec_name": audio_info.codec_name,
        "format_name": audio_info.format_name,
    }


def _source_metadata_path(input_path: Path) -> Path:
    return input_path.parent / ".metadata" / f"{input_path.stem}.json"


def _load_source_metadata(input_path: Path) -> dict[str, Any]:
    path = _source_metadata_path(input_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _prepared_metadata_path(output_path: Path) -> Path:
    return output_path.parent / ".metadata" / f"{output_path.stem}.json"


def _write_prepared_metadata(input_path: Path, output_path: Path, source_metadata: dict[str, Any]) -> None:
    payload = dict(source_metadata)
    payload.update(
        {
            "task_id": (source_metadata.get("task_id") or "").strip() or input_path.stem,
            "source_input_filename": input_path.name,
            "prepared_filename": output_path.name,
            "prepared_stem": output_path.stem,
        }
    )
    _write_json(_prepared_metadata_path(output_path), payload)


def _remove_prepared_metadata(output_path: Path) -> None:
    path = _prepared_metadata_path(output_path)
    if path.exists():
        path.unlink()


def _build_failure_payload(
    *,
    input_path: Path,
    output_path: Path,
    artifact_root: Path,
    requested_method: str,
    dry_run: bool,
    run_label: str,
    config_hash: str,
    error: Exception,
) -> dict[str, Any]:
    file_id = input_path.stem
    audio_info = _safe_probe_audio(input_path)
    source_metadata = _load_source_metadata(input_path)
    file_artifact_dir = artifact_root / "files" / file_id
    error_message = f"{type(error).__name__}: {error}"

    manifest_row = {
        "file_id": file_id,
        "original_path": str(input_path),
        "trimmed_path": "",
        "original_duration_sec": round(audio_info["duration_sec"], 3) if audio_info["duration_sec"] is not None else None,
        "trimmed_duration_sec": None,
        "leading_silence_removed_sec": None,
        "trailing_silence_removed_sec": None,
        "spoken_intro_removed_sec": None,
        "final_start_offset_sec": None,
        "final_end_offset_sec": None,
        "requested_method": requested_method,
        "method_chosen": "error",
        "predicted_intro_end_sec": None,
        "predicted_date_end_sec": None,
        "predicted_title_start_sec": None,
        "predicted_title_end_sec": None,
        "predicted_content_start_sec": None,
        "chosen_intro_trim_start_sec": None,
        "intro_overshoot_sec": None,
        "next_intro_boundary_sec": None,
        "opening_labels": "",
        "confidence_score": 0.0,
        "transcript_snippet": "",
        "failure_or_fallback_reason": error_message,
        "manual_review": True,
        "error_type": type(error).__name__,
    }

    payload = {
        "generated_at": _now_utc(),
        "run_label": run_label,
        "config_hash": config_hash,
        "dry_run": dry_run,
        "input": {
            "file_id": file_id,
            "path": str(input_path),
            "duration_sec": audio_info["duration_sec"],
            "sample_rate_hz": audio_info["sample_rate_hz"],
            "channels": audio_info["channels"],
            "codec_name": audio_info["codec_name"],
            "format_name": audio_info["format_name"],
        },
        "output": {
            "path": "",
            "artifacts_dir": str(file_artifact_dir),
            "debug_artifacts": {},
            "stale_output_removed": False,
        },
        "decision": {
            "requested_method": requested_method,
            "effective_method": "error",
            "manual_review": True,
            "manual_review_reasons": ["File preparation failed before output was emitted."],
            "reason": "File preparation failed before output was emitted.",
            "fallback_reason": error_message,
        },
        "manifest_row": manifest_row,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "intro_detection": None,
        "intro_trim": None,
        "transcript": None,
        "silence_backend": None,
        "vad_backend": None,
        "context_phrases": _derive_intro_context_phrases(file_id, source_metadata=source_metadata),
        "source_metadata": source_metadata,
    }

    if not dry_run and output_path.exists():
        output_path.unlink()
        _remove_prepared_metadata(output_path)
        payload["output"]["stale_output_removed"] = True

    _write_json(file_artifact_dir / f"{file_id}.trim.json", payload)
    return payload


def _backend_available(backend: Any) -> bool:
    available_method = getattr(backend, "available", None)
    if callable(available_method):
        try:
            return bool(available_method())
        except Exception:
            return False
    return True


def _backend_ready_reason(backend: Any, *, config: Any | None = None) -> str | None:
    check_ready = getattr(backend, "check_ready", None)
    if callable(check_ready):
        try:
            if config is not None:
                return check_ready(config)
            return check_ready()
        except TypeError:
            return check_ready()
        except Exception as exc:
            return str(exc)

    if _backend_available(backend):
        return None
    return "backend unavailable"


def _clean_context_source(value: str) -> str:
    source = Path(value.split("?", 1)[0]).stem
    source = _TRAILING_COPY_RE.sub("", source)
    return source.strip(" _-")


def _slug_context_phrase(value: str) -> str | None:
    cleaned = _clean_context_source(value)
    if not cleaned:
        return None

    parts = [part for part in cleaned.split("_") if part]
    if len(parts) < 3 or not _DATE_TOKEN_RE.match(parts[1]):
        return None

    relevant_parts = parts[2:-1] if len(parts) > 3 else parts[2:]
    words: list[str] = []
    for part in relevant_parts:
        for token in part.split("-"):
            token = token.strip()
            if not token or token.lower() in {"audio", "copy"}:
                continue
            words.append(token)

    phrase = " ".join(words).strip()
    return phrase or None


def _free_text_context_phrase(value: str) -> str | None:
    cleaned = _clean_context_source(value)
    if not cleaned:
        return None

    words = [
        token
        for token in re.split(r"[\s_-]+", cleaned)
        if token
        and token.lower() not in {"audio", "copy", "dp"}
        and not _DATE_TOKEN_RE.match(token)
    ]
    if not 1 <= len(words) <= 6:
        return None
    phrase = " ".join(words).strip()
    return phrase or None


def _derive_intro_context_phrases(file_id: str, *, source_metadata: dict[str, Any] | None = None) -> list[str]:
    phrases: list[str] = []

    def add_phrase(candidate: str | None) -> None:
        if not candidate:
            return
        normalized = candidate.strip()
        if normalized and normalized not in phrases:
            phrases.append(normalized)

    add_phrase(_slug_context_phrase(file_id))

    metadata = source_metadata or {}
    selected = metadata.get("selected_attachment") or {}
    for candidate in (
        selected.get("title"),
        selected.get("url_filename"),
        metadata.get("task_name"),
    ):
        add_phrase(_slug_context_phrase(candidate or ""))
        add_phrase(_free_text_context_phrase(candidate or ""))

    return phrases


class RawAudioPreparer:
    def __init__(
        self,
        config: TrimConfig,
        *,
        silence_backend: FfmpegSilenceBackend | None = None,
        vad_backend: WebRtcVadBackend | None = None,
        asr_backend: FasterWhisperAsrBackend | None = None,
    ) -> None:
        self.config = config
        self.silence_backend = silence_backend or FfmpegSilenceBackend()
        self.vad_backend = vad_backend or WebRtcVadBackend()
        self.asr_backend = asr_backend or FasterWhisperAsrBackend()

    def collect_inputs(self, input_dir: Path, file_names: list[str] | None = None) -> list[Path]:
        if not input_dir.exists():
            return []
        if file_names:
            return [input_dir / name for name in file_names]
        return sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )

    def _can_reuse_cached_sidecar(
        self,
        *,
        cached: dict[str, Any] | None,
        sidecar_path: Path,
        input_path: Path,
        output_path: Path,
        requested_method: str,
        dry_run: bool,
        config_hash: str,
    ) -> bool:
        if not cached:
            return False
        if cached.get("error") or (cached.get("manifest_row") or {}).get("method_chosen") == "error":
            return False
        if cached.get("config_hash") != config_hash:
            return False
        if sidecar_path.stat().st_mtime < input_path.stat().st_mtime:
            return False
        if not (dry_run or output_path.exists()):
            return False

        decision = cached.get("decision") or {}
        if decision.get("requested_method") != requested_method:
            return False

        if requested_method in {"ffmpeg-vad", "ffmpeg-vad-asr"} and _backend_available(self.vad_backend):
            cached_vad = bool((cached.get("vad_backend") or {}).get("available"))
            if not cached_vad:
                return False

        if requested_method == "ffmpeg-vad-asr" and _backend_available(self.asr_backend):
            cached_asr = bool(((cached.get("transcript") or {}).get("metadata") or {}).get("available"))
            if not cached_asr:
                return False

        return True

    def _preflight_requested_method(self, requested_method: str) -> None:
        if not self.config.decision.fail_on_backend_fallback:
            return

        failures: list[str] = []
        if requested_method in {"ffmpeg-vad", "ffmpeg-vad-asr"}:
            vad_reason = _backend_ready_reason(self.vad_backend)
            if vad_reason:
                failures.append(f"VAD backend not ready: {vad_reason}")

        if requested_method == "ffmpeg-vad-asr":
            asr_reason = _backend_ready_reason(self.asr_backend, config=self.config.asr)
            if asr_reason:
                failures.append(f"ASR backend not ready: {asr_reason}")

        if failures:
            joined = " | ".join(failures)
            python_bin = sys.executable
            raise BackendNotReadyError(
                f"Requested method '{requested_method}' cannot run with fallback disabled. "
                f"Python: {python_bin}. "
                f"{joined}. "
                f"If packages are missing, install them into this interpreter with: "
                f"'{python_bin} -m pip install webrtcvad-wheels faster-whisper'. "
                f"If ASR is installed but still not ready, make sure the configured Whisper model is available locally."
            )

    def _boundary_from_method(
        self,
        *,
        requested_method: str,
        duration_sec: float,
        silence: BoundaryEvidence,
        vad: BoundaryEvidence | None,
    ) -> tuple[str, float, float, float, str, list[str]]:
        reasons: list[str] = []
        effective_method = requested_method
        start = silence.start_offset_sec or 0.0
        end = silence.end_offset_sec or duration_sec
        confidence = silence.confidence
        reason = silence.reason

        if requested_method == "ffmpeg-silence":
            return effective_method, start, end, confidence, reason, reasons

        if not vad or not vad.available:
            reasons.append("VAD backend unavailable; falling back to ffmpeg-only trimming.")
            return "ffmpeg-silence", start, end, confidence, reason, reasons

        if (vad.confidence or 0.0) < self.config.vad.confidence_threshold:
            reasons.append("VAD confidence stayed below threshold; falling back to ffmpeg-only trimming.")
            return "ffmpeg-silence", start, end, confidence, reason, reasons

        silence_start = silence.start_offset_sec or 0.0
        silence_end = silence.end_offset_sec or duration_sec
        vad_start = vad.start_offset_sec or silence_start
        vad_end = vad.end_offset_sec or silence_end
        tolerance = self.config.vad.consensus_tolerance_sec
        start_delta = abs(vad_start - silence_start)
        end_delta = abs(vad_end - silence_end)

        if start_delta <= tolerance:
            start = max(silence_start, vad_start)
            start_reason = "ffmpeg and VAD agreed on the opening boundary."
        elif (
            vad_start > silence_start
            and (vad.confidence or 0.0) >= self.config.vad.confidence_threshold
            and silence_start <= self.config.decision.early_noise_guard_sec
        ):
            start = vad_start
            start_reason = "VAD overrode a very early non-speech burst near the file start."
            reasons.append("VAD start overrode a likely breath/noise onset before speech.")
        elif self.config.decision.safe_mode:
            start = min(silence_start, vad_start)
            start_reason = "Safe mode preferred the earlier opening boundary because ffmpeg and VAD disagreed."
            reasons.append("Opening boundary disagreement triggered safe-mode protection.")
        else:
            start = max(silence_start, vad_start)
            start_reason = "Unsafe mode preferred the later opening boundary during ffmpeg/VAD disagreement."

        if end_delta <= tolerance:
            end = min(silence_end, vad_end)
            end_reason = "ffmpeg and VAD agreed on the trailing boundary."
        elif self.config.decision.safe_mode:
            end = max(silence_end, vad_end)
            end_reason = "Safe mode preferred the later trailing boundary because ffmpeg and VAD disagreed."
            reasons.append("Trailing boundary disagreement triggered safe-mode protection.")
        else:
            end = min(silence_end, vad_end)
            end_reason = "Unsafe mode preferred the earlier trailing boundary during ffmpeg/VAD disagreement."

        confidence = round(
            max(0.0, min(0.99, ((silence.confidence or 0.0) + (vad.confidence or 0.0)) / 2 - max(start_delta, end_delta) * 0.08)),
            3,
        )
        reason = f"{start_reason} {end_reason}"
        return "ffmpeg-vad", start, end, confidence, reason, reasons

    def _next_word_start(self, transcript: TranscriptResult, trim_offset_sec: float) -> float | None:
        words = [word for segment in transcript.segments for word in segment.words]
        next_word = next((word for word in words if word.start_sec >= trim_offset_sec + 0.001), None)
        if next_word is None:
            return None
        return next_word.start_sec

    def _next_intro_boundary(self, intro: IntroDetectionResult, transcript: TranscriptResult) -> tuple[float | None, str | None]:
        evidence = intro.evidence or {}
        next_preserved_start = evidence.get("next_preserved_start_sec")
        next_preserved_label = evidence.get("next_preserved_label")
        next_word_start = evidence.get("next_word_start_sec")
        if next_word_start is None and intro.trim_offset_sec is not None:
            next_word_start = self._next_word_start(transcript, intro.trim_offset_sec)

        candidates: list[tuple[float, str]] = []
        if next_preserved_start is not None:
            label = str(next_preserved_label or "preserved_segment")
            candidates.append((float(next_preserved_start), label))
        if next_word_start is not None:
            candidates.append((float(next_word_start), "word"))
        if not candidates:
            return None, None
        return min(candidates, key=lambda item: item[0])

    def _intro_pad(self, transcript: TranscriptResult, trim_offset_sec: float, *, next_boundary_sec: float | None = None) -> float:
        desired = max(
            self.config.intro_classifier.post_intro_pad_sec,
            self.config.intro_classifier.min_date_overshoot_sec,
        )
        if desired <= 0:
            return 0.0

        boundary_sec = next_boundary_sec
        if boundary_sec is None:
            boundary_sec = self._next_word_start(transcript, trim_offset_sec)
        if boundary_sec is None:
            return desired

        latest_reasonable_start = boundary_sec + _MAX_NEXT_BOUNDARY_INTRUSION_SEC
        return max(0.0, min(desired, latest_reasonable_start - trim_offset_sec))

    def _plan_intro_trim(self, intro: IntroDetectionResult, transcript: TranscriptResult) -> dict[str, Any]:
        if not intro.detected or intro.trim_offset_sec is None:
            return {
                "detected": False,
                "date_end_sec": None,
                "requested_overshoot_sec": 0.0,
                "overshoot_used_sec": 0.0,
                "requested_trim_start_sec": None,
                "chosen_trim_start_sec": None,
                "next_word_start_sec": None,
                "next_preserved_start_sec": None,
                "next_preserved_label": None,
                "next_intro_boundary_sec": None,
                "next_intro_boundary_kind": None,
                "gap_to_next_boundary_sec": None,
                "clamped_to_next_boundary": False,
                "manual_review_reasons": [],
                "manual_review_clearance": "No spoken date was detected.",
            }

        evidence = intro.evidence or {}
        requested_overshoot = max(
            self.config.intro_classifier.post_intro_pad_sec,
            self.config.intro_classifier.min_date_overshoot_sec,
        )
        next_word_start = evidence.get("next_word_start_sec")
        if next_word_start is None:
            next_word_start = self._next_word_start(transcript, intro.trim_offset_sec)
        next_preserved_start = evidence.get("next_preserved_start_sec")
        next_preserved_label = evidence.get("next_preserved_label")
        next_boundary_sec, next_boundary_kind = self._next_intro_boundary(intro, transcript)
        overshoot_used = self._intro_pad(
            transcript,
            intro.trim_offset_sec,
            next_boundary_sec=next_boundary_sec,
        )
        requested_trim_start = intro.trim_offset_sec + requested_overshoot
        chosen_trim_start = intro.trim_offset_sec + overshoot_used
        gap_to_next_boundary = None
        if next_boundary_sec is not None:
            gap_to_next_boundary = max(0.0, next_boundary_sec - intro.trim_offset_sec)

        review_reasons: list[str] = []
        if next_boundary_sec is None:
            review_reasons.append("Spoken date was detected but no reliable next word/content boundary was available.")
        elif gap_to_next_boundary is not None and gap_to_next_boundary < self.config.intro_classifier.min_date_overshoot_sec:
            review_reasons.append("Spoken date ended extremely close to the next preserved speech boundary.")

        if not any(segment.label in {"title", "content"} for segment in intro.classified_segments):
            review_reasons.append("Spoken date was detected but the following title/content segmentation was missing.")

        if overshoot_used + 0.001 < self.config.intro_classifier.min_date_overshoot_sec:
            review_reasons.append("Automatic spoken-date overshoot stayed below the safe minimum threshold.")

        return {
            "detected": True,
            "date_end_sec": round(intro.trim_offset_sec, 3),
            "requested_overshoot_sec": round(requested_overshoot, 3),
            "overshoot_used_sec": round(overshoot_used, 3),
            "requested_trim_start_sec": round(requested_trim_start, 3),
            "chosen_trim_start_sec": round(chosen_trim_start, 3),
            "next_word_start_sec": round(float(next_word_start), 3) if next_word_start is not None else None,
            "next_preserved_start_sec": round(float(next_preserved_start), 3) if next_preserved_start is not None else None,
            "next_preserved_label": next_preserved_label,
            "next_intro_boundary_sec": round(float(next_boundary_sec), 3) if next_boundary_sec is not None else None,
            "next_intro_boundary_kind": next_boundary_kind,
            "gap_to_next_boundary_sec": round(gap_to_next_boundary, 3) if gap_to_next_boundary is not None else None,
            "clamped_to_next_boundary": chosen_trim_start + 0.001 < requested_trim_start,
            "manual_review_reasons": review_reasons,
            "manual_review_clearance": (
                "No opening ambiguity was detected around the spoken-date cut."
                if not review_reasons
                else ""
            ),
        }

    def prepare_file(
        self,
        input_path: Path,
        *,
        output_dir: Path,
        artifact_root: Path,
        requested_method: str,
        dry_run: bool = False,
        debug: bool = False,
        force: bool = False,
        skip_existing: bool = False,
        run_label: str = "production",
    ) -> dict[str, Any]:
        if requested_method not in METHODS:
            raise ValueError(f"Unsupported method: {requested_method}")
        self._preflight_requested_method(requested_method)

        file_id = input_path.stem
        file_artifact_dir = artifact_root / "files" / file_id
        sidecar_path = file_artifact_dir / f"{file_id}.trim.json"
        output_path = output_dir / f"{file_id}.wav"
        config_hash = self.config.fingerprint()
        source_metadata = _load_source_metadata(input_path)

        if skip_existing and not force:
            cached = _load_cached_sidecar(sidecar_path)
            existing_is_fresh = self._can_reuse_cached_sidecar(
                cached=cached,
                sidecar_path=sidecar_path,
                input_path=input_path,
                output_path=output_path,
                requested_method=requested_method,
                dry_run=dry_run,
                config_hash=config_hash,
            )
            if existing_is_fresh:
                if not dry_run and output_path.exists():
                    _write_prepared_metadata(input_path, output_path, source_metadata)
                cached["manifest_row"]["skipped_existing"] = True
                return cached

        audio_info = probe_audio(input_path)
        silence = self.silence_backend.analyze(
            input_path,
            duration_sec=audio_info.duration_sec,
            config=self.config.silence,
        )
        vad = None
        if requested_method in {"ffmpeg-vad", "ffmpeg-vad-asr"} and self.config.vad.enabled:
            vad = self.vad_backend.analyze(
                input_path,
                duration_sec=audio_info.duration_sec,
                config=self.config.vad,
            )

        transcript = TranscriptResult(backend=self.asr_backend.name)
        if requested_method == "ffmpeg-vad-asr" and self.config.asr.enabled:
            transcript = self.asr_backend.transcribe(input_path, config=self.config.asr)
        context_phrases = _derive_intro_context_phrases(file_id, source_metadata=source_metadata)

        intro = IntroDetectionResult(
            detected=False,
            confidence=0.0,
            reason="ASR intro analysis was not requested.",
            transcript_snippet=_transcript_snippet(transcript),
        )
        if requested_method == "ffmpeg-vad-asr" and self.config.intro_classifier.enabled:
            intro = detect_intro(
                transcript,
                config=self.config.intro_classifier,
                context_phrases=context_phrases,
            )
            _validate_intro_detection(intro)

        boundary_method, boundary_start, boundary_end, boundary_confidence, boundary_reason, fallback_reasons = self._boundary_from_method(
            requested_method=requested_method,
            duration_sec=audio_info.duration_sec,
            silence=silence,
            vad=vad,
        )
        effective_method = boundary_method
        asr_available = bool(transcript.metadata.get("available")) if transcript.metadata else False
        if requested_method == "ffmpeg-vad-asr":
            if boundary_method == "ffmpeg-vad" and asr_available:
                effective_method = "ffmpeg-vad-asr"
            elif not asr_available:
                asr_reason = (transcript.metadata or {}).get("reason") or "unknown ASR failure"
                fallback_reasons.append(f"ASR backend unavailable; falling back to boundary-only trimming. Reason: {asr_reason}")

        manual_review_reasons: list[str] = []
        final_start = boundary_start
        final_end = boundary_end
        spoken_intro_removed_sec = 0.0
        intro_trim = self._plan_intro_trim(intro, transcript)

        if silence.confidence <= 0.10 and (not vad or vad.confidence <= 0.10):
            fallback_reasons.append("No speech window could be isolated with high confidence.")
            manual_review_reasons.append("No speech window could be isolated.")
            final_start = 0.0
            final_end = audio_info.duration_sec

        if requested_method == "ffmpeg-vad-asr" and asr_available and intro.detected and intro.trim_offset_sec is not None:
            intro_start = float(intro_trim["chosen_trim_start_sec"])
            manual_review_reasons.extend(intro_trim["manual_review_reasons"])
            if self.config.decision.safe_mode and intro_start > self.config.decision.max_start_trim_sec:
                fallback_reasons.append("Detected intro exceeded the maximum safe automatic start trim.")
                manual_review_reasons.append("Intro trim exceeded maximum safe automatic start trim.")
            else:
                final_start = max(final_start, intro_start)
                spoken_intro_removed_sec = max(0.0, final_start - boundary_start)

        if requested_method == "ffmpeg-vad-asr" and effective_method != "ffmpeg-vad-asr":
            manual_review_reasons.append("Advanced method fell back because an optional backend was unavailable or low-confidence.")

        if final_start > self.config.decision.max_start_trim_sec:
            manual_review_reasons.append("Opening trim exceeded the configured maximum safe start trim.")
            if self.config.decision.safe_mode:
                final_start = min(boundary_start, self.config.decision.max_start_trim_sec)

        if final_end - final_start < self.config.decision.min_output_duration_sec:
            fallback_reasons.append("Trimmed audio would be too short; keeping original window for safety.")
            manual_review_reasons.append("Trimmed audio would be too short.")
            final_start = 0.0
            final_end = audio_info.duration_sec
            spoken_intro_removed_sec = 0.0

        leading_silence_removed_sec = max(0.0, min(final_start, boundary_start))
        trailing_silence_removed_sec = max(0.0, audio_info.duration_sec - final_end)
        confidence = boundary_confidence
        if effective_method == "ffmpeg-vad-asr" and intro.detected:
            confidence = round(min(0.99, (boundary_confidence * 0.6) + (intro.confidence * 0.4)), 3)

        if confidence < self.config.decision.manual_review_confidence_below:
            manual_review_reasons.append("Overall trim confidence fell below the manual-review threshold.")

        manual_review_reasons = list(dict.fromkeys(manual_review_reasons))
        manual_review = bool(manual_review_reasons)
        decision_reason = boundary_reason
        if spoken_intro_removed_sec > 0:
            decision_reason = f"{decision_reason} The opening spoken date was removed while preserving the title and content."

        decision = DecisionRecord(
            requested_method=requested_method,
            effective_method=effective_method,
            start_offset_sec=round(final_start, 3),
            end_offset_sec=round(final_end, 3),
            leading_silence_removed_sec=round(leading_silence_removed_sec, 3),
            trailing_silence_removed_sec=round(trailing_silence_removed_sec, 3),
            spoken_intro_removed_sec=round(spoken_intro_removed_sec, 3),
            confidence=round(confidence, 3),
            reason=decision_reason,
            fallback_reason=" | ".join(fallback_reasons) if fallback_reasons else None,
            manual_review=manual_review,
            manual_review_reasons=manual_review_reasons,
        )

        debug_artifacts: dict[str, Any] = {}
        if debug:
            waveform_path = file_artifact_dir / f"{file_id}.waveform.png"
            render_waveform_png(input_path, waveform_path)
            debug_artifacts["waveform_png"] = str(waveform_path)
            if transcript.text:
                transcript_path = file_artifact_dir / f"{file_id}.intro.txt"
                transcript_path.write_text(transcript.text + "\n", encoding="utf-8")
                debug_artifacts["transcript_preview"] = str(transcript_path)

        if not dry_run:
            trim_audio_to_wav(
                input_path,
                output_path,
                start_offset_sec=final_start,
                end_offset_sec=final_end,
            )
            _write_prepared_metadata(input_path, output_path, source_metadata)

        manifest_row = {
            "file_id": file_id,
            "original_path": str(input_path),
            "trimmed_path": "" if dry_run else str(output_path),
            "original_duration_sec": round(audio_info.duration_sec, 3),
            "trimmed_duration_sec": round(final_end - final_start, 3),
            "leading_silence_removed_sec": round(leading_silence_removed_sec, 3),
            "trailing_silence_removed_sec": round(trailing_silence_removed_sec, 3),
            "spoken_intro_removed_sec": round(spoken_intro_removed_sec, 3),
            "final_start_offset_sec": round(final_start, 3),
            "final_end_offset_sec": round(final_end, 3),
            "requested_method": requested_method,
            "method_chosen": effective_method,
            "predicted_intro_end_sec": round(intro.trim_offset_sec, 3) if intro.trim_offset_sec is not None else "",
            "predicted_date_end_sec": round((intro.evidence or {}).get("date_end_sec"), 3) if (intro.evidence or {}).get("date_end_sec") is not None else "",
            "predicted_title_start_sec": round(_classified_span_bounds(intro, "title")[0], 3) if _classified_span_bounds(intro, "title")[0] is not None else "",
            "predicted_title_end_sec": round(_classified_span_bounds(intro, "title")[1], 3) if _classified_span_bounds(intro, "title")[1] is not None else "",
            "predicted_content_start_sec": round(_classified_span_bounds(intro, "content")[0], 3) if _classified_span_bounds(intro, "content")[0] is not None else "",
            "chosen_intro_trim_start_sec": intro_trim["chosen_trim_start_sec"] if intro.detected else "",
            "intro_overshoot_sec": intro_trim["overshoot_used_sec"] if intro.detected else "",
            "next_intro_boundary_sec": intro_trim["next_intro_boundary_sec"] if intro.detected else "",
            "opening_labels": _opening_labels(intro),
            "confidence_score": round(confidence, 3),
            "transcript_snippet": intro.transcript_snippet or _transcript_snippet(transcript),
            "failure_or_fallback_reason": decision.fallback_reason or "",
            "manual_review": manual_review,
        }

        payload = {
            "generated_at": _now_utc(),
            "run_label": run_label,
            "config_hash": config_hash,
            "dry_run": dry_run,
            "input": {
                "file_id": file_id,
                "path": str(input_path),
                "duration_sec": audio_info.duration_sec,
                "sample_rate_hz": audio_info.sample_rate_hz,
                "channels": audio_info.channels,
                "codec_name": audio_info.codec_name,
                "format_name": audio_info.format_name,
            },
            "output": {
                "path": "" if dry_run else str(output_path),
                "artifacts_dir": str(file_artifact_dir),
                "debug_artifacts": debug_artifacts,
            },
            "decision": _json_ready(decision),
            "manifest_row": manifest_row,
            "silence_backend": _json_ready(silence),
            "vad_backend": _json_ready(vad) if vad else None,
            "transcript": _json_ready(transcript),
            "intro_detection": _json_ready(intro),
            "intro_trim": intro_trim,
            "context_phrases": context_phrases,
            "source_metadata": source_metadata,
        }
        _write_json(sidecar_path, payload)
        return payload

    def prepare_batch(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        artifact_root: Path,
        requested_method: str,
        file_names: list[str] | None = None,
        dry_run: bool = False,
        debug: bool = False,
        force: bool = False,
        skip_existing: bool = False,
        run_label: str = "production",
    ) -> dict[str, Any]:
        input_paths = self.collect_inputs(input_dir, file_names=file_names)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        sidecars: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for input_path in input_paths:
            output_path = output_dir / f"{input_path.stem}.wav"
            try:
                sidecar = self.prepare_file(
                    input_path,
                    output_dir=output_dir,
                    artifact_root=artifact_root,
                    requested_method=requested_method,
                    dry_run=dry_run,
                    debug=debug,
                    force=force,
                    skip_existing=skip_existing,
                    run_label=run_label,
                )
                events.append(
                    {
                        "event": "file_prepared",
                        "generated_at": _now_utc(),
                        "requested_method": requested_method,
                        **sidecar["manifest_row"],
                    }
                )
            except BackendNotReadyError:
                raise
            except Exception as exc:
                sidecar = _build_failure_payload(
                    input_path=input_path,
                    output_path=output_path,
                    artifact_root=artifact_root,
                    requested_method=requested_method,
                    dry_run=dry_run,
                    run_label=run_label,
                    config_hash=self.config.fingerprint(),
                    error=exc,
                )
                events.append(
                    {
                        "event": "file_failed",
                        "generated_at": _now_utc(),
                        "requested_method": requested_method,
                        **sidecar["manifest_row"],
                    }
                )
            sidecars.append(sidecar)
            rows.append(sidecar["manifest_row"])

        manifest_jsonl = artifact_root / "manifest.jsonl"
        manifest_csv = artifact_root / "manifest.csv"
        events_jsonl = artifact_root / "events.jsonl"
        summary_path = artifact_root / "summary.json"

        _write_jsonl(manifest_jsonl, rows)
        _write_csv(manifest_csv, rows)
        _write_jsonl(events_jsonl, events)

        files_succeeded = sum(1 for row in rows if row.get("method_chosen") != "error")
        files_failed = sum(1 for row in rows if row.get("method_chosen") == "error")

        summary = {
            "generated_at": _now_utc(),
            "run_label": run_label,
            "requested_method": requested_method,
            "files_seen": len(input_paths),
            "files_emitted": files_succeeded,
            "files_failed": files_failed,
            "manual_review_count": sum(1 for row in rows if row.get("manual_review")),
            "dry_run": dry_run,
            "output_dir": str(output_dir),
            "artifact_root": str(artifact_root),
            "events_jsonl": str(events_jsonl),
            "manifest_jsonl": str(manifest_jsonl),
            "manifest_csv": str(manifest_csv),
        }
        _write_json(summary_path, summary)
        return {
            "summary": summary,
            "rows": rows,
            "sidecars": sidecars,
        }
