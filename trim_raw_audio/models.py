from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AudioInfo:
    path: Path
    duration_sec: float
    sample_rate_hz: int
    channels: int
    codec_name: str
    format_name: str


@dataclass(slots=True)
class Span:
    start_sec: float
    end_sec: float
    label: str
    confidence: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(slots=True)
class TranscriptWord:
    word: str
    start_sec: float
    end_sec: float


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    start_sec: float
    end_sec: float
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptResult:
    text: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    confidence: float | None = None
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BoundaryEvidence:
    backend: str
    available: bool
    start_offset_sec: float | None = None
    end_offset_sec: float | None = None
    speech_start_sec: float | None = None
    speech_end_sec: float | None = None
    confidence: float = 0.0
    reason: str = ""
    spans: list[Span] = field(default_factory=list)
    tool_output: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntroSegmentClassification:
    label: str
    start_sec: float
    end_sec: float
    text: str
    confidence: float = 0.0
    removable: bool = False
    matched_patterns: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class IntroDetectionResult:
    detected: bool
    trim_offset_sec: float | None = None
    confidence: float = 0.0
    reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)
    transcript_snippet: str = ""
    classified_segments: list[IntroSegmentClassification] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionRecord:
    requested_method: str
    effective_method: str
    start_offset_sec: float
    end_offset_sec: float
    leading_silence_removed_sec: float
    trailing_silence_removed_sec: float
    spoken_intro_removed_sec: float
    confidence: float
    reason: str
    fallback_reason: str | None = None
    manual_review: bool = False
    manual_review_reasons: list[str] = field(default_factory=list)
