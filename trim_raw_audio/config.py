from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathsConfig:
    input_dir: str = "audio-input"
    output_dir: str = "audio-prepared"
    artifact_dir: str = "trim-artifacts"
    experiment_dir: str = "trim-experiments"


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    structured: bool = True


@dataclass(slots=True)
class SilenceConfig:
    min_silence_duration_sec: float = 0.35
    noise_threshold_db: float = -35.0
    start_pad_sec: float = 0.12
    end_pad_sec: float = 0.20


@dataclass(slots=True)
class VadConfig:
    enabled: bool = True
    backend: str = "webrtcvad"
    aggressiveness: int = 2
    frame_ms: int = 30
    confidence_threshold: float = 0.60
    start_pad_sec: float = 0.05
    end_pad_sec: float = 0.20
    min_speech_duration_sec: float = 0.25
    max_merge_gap_sec: float = 0.30
    consensus_tolerance_sec: float = 0.18


@dataclass(slots=True)
class AsrConfig:
    enabled: bool = True
    backend: str = "faster_whisper"
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    analysis_window_sec: float = 20.0
    language: str | None = None


@dataclass(slots=True)
class IntroClassifierConfig:
    enabled: bool = True
    confidence_threshold: float = 0.74
    date_segment_min_score: float = 0.18
    post_intro_pad_sec: float = 0.15
    min_date_overshoot_sec: float = 0.22
    max_date_overshoot_sec: float = 0.40
    max_auto_trim_start_sec: float = 12.0
    title_confidence_threshold: float = 0.62
    title_max_words: int = 6
    title_max_duration_sec: float = 3.5
    boundary_snap_window_sec: float = 0.18
    boundary_snap_frame_ms: int = 5
    boundary_snap_threshold_db: float = -35.0
    no_intro_start_preroll_sec: float = 0.20
    no_intro_max_backward_snap_sec: float = 0.15
    leading_fade_in_ms: int = 15
    repeated_phrases: list[str] = field(default_factory=list)
    boilerplate_phrases: list[str] = field(
        default_factory=lambda: [
            "today is",
            "this is",
            "recording for",
            "recorded on",
            "sacred space",
            "voice note",
            "journal entry",
        ]
    )


@dataclass(slots=True)
class DecisionConfig:
    safe_mode: bool = True
    fail_on_backend_fallback: bool = True
    min_output_duration_sec: float = 0.50
    manual_review_confidence_below: float = 0.65
    early_noise_guard_sec: float = 0.35
    max_start_trim_sec: float = 15.0


@dataclass(slots=True)
class ExperimentConfig:
    default_methods: list[str] = field(
        default_factory=lambda: ["ffmpeg-silence", "ffmpeg-vad", "ffmpeg-vad-asr"]
    )
    shortlist_size: int = 10
    sample_size: int = 12


@dataclass(slots=True)
class TrimConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    intro_classifier: IntroClassifierConfig = field(default_factory=IntroClassifierConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _apply_section(section_obj: Any, payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    for key, value in payload.items():
        if hasattr(section_obj, key):
            setattr(section_obj, key, value)


def load_trim_config(path: Path | None = None) -> TrimConfig:
    config = TrimConfig()
    if not path:
        return config
    if not path.exists():
        return config

    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    _apply_section(config.paths, payload.get("paths"))
    _apply_section(config.logging, payload.get("logging"))
    _apply_section(config.silence, payload.get("silence"))
    _apply_section(config.vad, payload.get("vad"))
    _apply_section(config.asr, payload.get("asr"))
    _apply_section(config.intro_classifier, payload.get("intro_classifier"))
    _apply_section(config.decision, payload.get("decision"))
    _apply_section(config.experiment, payload.get("experiment"))
    return config
