from __future__ import annotations

import math
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from .audio import extract_pcm_mono_16k, render_excerpt_for_asr, run_silencedetect
from .config import AsrConfig, SilenceConfig, VadConfig
from .models import BoundaryEvidence, Span, TranscriptResult, TranscriptSegment, TranscriptWord

try:
    import webrtcvad as _webrtcvad
except Exception as exc:  # pragma: no cover - exercised indirectly
    _webrtcvad = None
    _WEBRTCVAD_IMPORT_ERROR = str(exc)
else:  # pragma: no cover - exercised indirectly
    _WEBRTCVAD_IMPORT_ERROR = None

try:
    from faster_whisper import WhisperModel
except Exception as exc:  # pragma: no cover - exercised indirectly
    WhisperModel = None
    _FASTER_WHISPER_IMPORT_ERROR = str(exc)
else:  # pragma: no cover - exercised indirectly
    _FASTER_WHISPER_IMPORT_ERROR = None


_SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<value>-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*(?P<value>\d+(?:\.\d+)?)\s*\|\s*silence_duration:\s*(?P<duration>\d+(?:\.\d+)?)"
)


def _parse_silencedetect(log_text: str) -> list[dict[str, float | str]]:
    events: list[dict[str, float | str]] = []
    for line in log_text.splitlines():
        if match := _SILENCE_START_RE.search(line):
            events.append({"type": "start", "time": float(match.group("value"))})
        elif match := _SILENCE_END_RE.search(line):
            events.append(
                {
                    "type": "end",
                    "time": float(match.group("value")),
                    "duration": float(match.group("duration")),
                }
            )
    return events


def _leading_silence_end(events: list[dict[str, float | str]], duration_sec: float) -> float:
    for index, event in enumerate(events):
        if event["type"] != "start":
            continue
        time_value = float(event.get("time") or 0.0)
        if abs(time_value) > 0.05:
            continue
        for later in events[index + 1 :]:
            if later["type"] == "end":
                return float(later.get("time") or 0.0)
        return duration_sec
    return 0.0


class FfmpegSilenceBackend:
    name = "ffmpeg_silencedetect"

    def analyze(
        self,
        input_path: Path,
        *,
        duration_sec: float,
        config: SilenceConfig,
    ) -> BoundaryEvidence:
        forward_log = run_silencedetect(
            input_path,
            noise_threshold_db=config.noise_threshold_db,
            min_silence_duration_sec=config.min_silence_duration_sec,
            reverse=False,
        )
        reverse_log = run_silencedetect(
            input_path,
            noise_threshold_db=config.noise_threshold_db,
            min_silence_duration_sec=config.min_silence_duration_sec,
            reverse=True,
        )
        forward_events = _parse_silencedetect(forward_log)
        reverse_events = _parse_silencedetect(reverse_log)
        speech_start = _leading_silence_end(forward_events, duration_sec)
        reversed_leading_silence = _leading_silence_end(reverse_events, duration_sec)
        speech_end = max(speech_start, duration_sec - reversed_leading_silence)

        no_speech_detected = speech_end - speech_start <= 0.01
        if no_speech_detected:
            return BoundaryEvidence(
                backend=self.name,
                available=True,
                start_offset_sec=0.0,
                end_offset_sec=duration_sec,
                speech_start_sec=None,
                speech_end_sec=None,
                confidence=0.05,
                reason="No non-silent region detected; keeping original audio for safety.",
                spans=[],
                tool_output={
                    "forward_events": forward_events,
                    "reverse_events": reverse_events,
                    "leading_silence_end_sec": speech_start,
                    "trailing_silence_duration_sec": reversed_leading_silence,
                },
            )

        start_offset = max(0.0, speech_start - config.start_pad_sec)
        end_offset = min(duration_sec, speech_end + config.end_pad_sec)
        return BoundaryEvidence(
            backend=self.name,
            available=True,
            start_offset_sec=start_offset,
            end_offset_sec=end_offset,
            speech_start_sec=speech_start,
            speech_end_sec=speech_end,
            confidence=0.82,
            reason="Boundary silence estimated with ffmpeg silencedetect.",
            spans=[
                Span(
                    start_sec=speech_start,
                    end_sec=speech_end,
                    label="speech_window",
                    confidence=0.82,
                    source=self.name,
                )
            ],
            tool_output={
                "forward_events": forward_events,
                "reverse_events": reverse_events,
                "leading_silence_end_sec": speech_start,
                "trailing_silence_duration_sec": reversed_leading_silence,
            },
        )


class WebRtcVadBackend:
    name = "webrtcvad"

    def available(self) -> bool:
        return _webrtcvad is not None

    def check_ready(self) -> str | None:
        if self.available():
            return None
        return f"webrtcvad unavailable: {_WEBRTCVAD_IMPORT_ERROR}"

    def analyze(
        self,
        input_path: Path,
        *,
        duration_sec: float,
        config: VadConfig,
    ) -> BoundaryEvidence:
        if not self.available():
            return BoundaryEvidence(
                backend=self.name,
                available=False,
                confidence=0.0,
                reason=f"webrtcvad unavailable: {_WEBRTCVAD_IMPORT_ERROR}",
            )

        pcm = extract_pcm_mono_16k(input_path)
        sample_rate = 16000
        frame_ms = config.frame_ms
        frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
        if frame_bytes <= 0:
            raise ValueError("VAD frame size must be positive")

        vad = _webrtcvad.Vad(max(0, min(3, config.aggressiveness)))
        raw_spans: list[Span] = []
        total_frames = 0

        current_start: float | None = None
        current_end: float | None = None

        for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[offset : offset + frame_bytes]
            frame_start = offset / 2 / sample_rate
            frame_end = frame_start + frame_ms / 1000.0
            total_frames += 1
            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech:
                if current_start is None:
                    current_start = frame_start
                current_end = frame_end
            elif current_start is not None and current_end is not None:
                raw_spans.append(
                    Span(
                        start_sec=current_start,
                        end_sec=current_end,
                        label="speech",
                        confidence=0.80,
                        source=self.name,
                    )
                )
                current_start = None
                current_end = None

        if current_start is not None and current_end is not None:
            raw_spans.append(
                Span(
                    start_sec=current_start,
                    end_sec=current_end,
                    label="speech",
                    confidence=0.80,
                    source=self.name,
                )
            )

        merged_spans: list[Span] = []
        for span in raw_spans:
            if not merged_spans:
                merged_spans.append(span)
                continue
            previous = merged_spans[-1]
            if span.start_sec - previous.end_sec <= config.max_merge_gap_sec:
                previous.end_sec = span.end_sec
            else:
                merged_spans.append(span)

        speech_spans = [
            span for span in merged_spans if span.duration_sec >= config.min_speech_duration_sec
        ]

        if not speech_spans:
            return BoundaryEvidence(
                backend=self.name,
                available=True,
                confidence=0.10,
                reason="No VAD speech span crossed the minimum duration threshold.",
                tool_output={
                    "total_frames": total_frames,
                    "raw_spans": [asdict(span) for span in raw_spans],
                },
            )

        speech_total = sum(span.duration_sec for span in speech_spans)
        speech_ratio = min(1.0, speech_total / max(duration_sec, 0.001))
        confidence = 0.60
        confidence += min(0.18, speech_total / 8.0)
        confidence += min(0.12, speech_ratio)
        confidence = round(min(confidence, 0.95), 3)

        speech_start = speech_spans[0].start_sec
        speech_end = speech_spans[-1].end_sec
        return BoundaryEvidence(
            backend=self.name,
            available=True,
            start_offset_sec=max(0.0, speech_start - config.start_pad_sec),
            end_offset_sec=min(duration_sec, speech_end + config.end_pad_sec),
            speech_start_sec=speech_start,
            speech_end_sec=speech_end,
            confidence=confidence,
            reason="Speech boundaries estimated with WebRTC VAD.",
            spans=speech_spans,
            tool_output={
                "total_frames": total_frames,
                "speech_ratio": round(speech_ratio, 4),
                "raw_spans": [asdict(span) for span in raw_spans],
            },
        )


class FasterWhisperAsrBackend:
    name = "faster_whisper"

    def __init__(self) -> None:
        self._model_cache: dict[tuple[str, str, str], WhisperModel] = {}

    def available(self) -> bool:
        return WhisperModel is not None

    def check_ready(self, config: AsrConfig) -> str | None:
        if not self.available():
            return _FASTER_WHISPER_IMPORT_ERROR or "faster-whisper unavailable"
        try:
            self._get_model(config)
        except Exception as exc:
            return str(exc)
        return None

    def _get_model(self, config: AsrConfig) -> WhisperModel:
        key = (config.model_size, config.device, config.compute_type)
        if key not in self._model_cache:
            if WhisperModel is None:
                raise RuntimeError(_FASTER_WHISPER_IMPORT_ERROR or "faster-whisper unavailable")
            self._model_cache[key] = WhisperModel(
                config.model_size,
                device=config.device,
                compute_type=config.compute_type,
            )
        return self._model_cache[key]

    def transcribe(self, input_path: Path, *, config: AsrConfig) -> TranscriptResult:
        if not self.available():
            return TranscriptResult(
                backend=self.name,
                metadata={"available": False, "reason": _FASTER_WHISPER_IMPORT_ERROR},
            )

        try:
            with tempfile.TemporaryDirectory(prefix="trim-raw-audio-") as temp_dir:
                excerpt_path = Path(temp_dir) / "intro-window.wav"
                render_excerpt_for_asr(
                    input_path,
                    excerpt_path,
                    analysis_window_sec=config.analysis_window_sec,
                )

                model = self._get_model(config)
                segments, info = model.transcribe(
                    str(excerpt_path),
                    language=config.language,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    vad_filter=False,
                )

                transcript_segments: list[TranscriptSegment] = []
                all_text_parts: list[str] = []
                confidence_parts: list[float] = []
                for segment in segments:
                    words = [
                        TranscriptWord(
                            word=(word.word or "").strip(),
                            start_sec=float(word.start or segment.start),
                            end_sec=float(word.end or segment.end),
                        )
                        for word in (segment.words or [])
                        if (word.word or "").strip()
                    ]
                    transcript_segments.append(
                        TranscriptSegment(
                            text=(segment.text or "").strip(),
                            start_sec=float(segment.start),
                            end_sec=float(segment.end),
                            avg_logprob=getattr(segment, "avg_logprob", None),
                            no_speech_prob=getattr(segment, "no_speech_prob", None),
                            words=words,
                        )
                    )
                    if segment.text:
                        all_text_parts.append(segment.text.strip())
                    no_speech_prob = getattr(segment, "no_speech_prob", None)
                    if no_speech_prob is not None:
                        confidence_parts.append(max(0.0, 1.0 - float(no_speech_prob)))
        except Exception as exc:
            return TranscriptResult(
                backend=self.name,
                metadata={
                    "available": False,
                    "reason": str(exc),
                    "analysis_window_sec": config.analysis_window_sec,
                    "model_size": config.model_size,
                },
            )

        confidence = None
        if confidence_parts:
            confidence = round(sum(confidence_parts) / len(confidence_parts), 3)

        return TranscriptResult(
            text=" ".join(part for part in all_text_parts if part).strip(),
            segments=transcript_segments,
            language=getattr(info, "language", None),
            confidence=confidence,
            backend=self.name,
            metadata={
                "available": True,
                "analysis_window_sec": config.analysis_window_sec,
                "model_size": config.model_size,
            },
        )
