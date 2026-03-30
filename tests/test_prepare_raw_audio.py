from __future__ import annotations

import json
import math
import wave
from array import array
from pathlib import Path

import pytest

import trim_raw_audio.pipeline as pipeline_module
from trim_raw_audio import RawAudioPreparer, load_trim_config
from trim_raw_audio.models import (
    BoundaryEvidence,
    IntroDetectionResult,
    IntroSegmentClassification,
    Span,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
)


SAMPLE_RATE = 16_000


def _silence(duration_sec: float) -> array:
    return array("h", [0] * int(SAMPLE_RATE * duration_sec))


def _tone(duration_sec: float, *, frequency_hz: float = 220.0, amplitude: float = 0.4) -> array:
    total = int(SAMPLE_RATE * duration_sec)
    return array(
        "h",
        [
            int(32767 * amplitude * math.sin(2.0 * math.pi * frequency_hz * (index / SAMPLE_RATE)))
            for index in range(total)
        ],
    )


def _decay_tone(
    duration_sec: float,
    *,
    frequency_hz: float = 220.0,
    start_amplitude: float = 0.20,
    end_amplitude: float = 0.0,
) -> array:
    total = int(SAMPLE_RATE * duration_sec)
    if total <= 0:
        return array("h")
    return array(
        "h",
        [
            int(
                32767
                * (start_amplitude + ((end_amplitude - start_amplitude) * (index / max(total - 1, 1))))
                * math.sin(2.0 * math.pi * frequency_hz * (index / SAMPLE_RATE))
            )
            for index in range(total)
        ],
    )


def _noise(duration_sec: float, *, amplitude: float = 0.03) -> array:
    total = int(SAMPLE_RATE * duration_sec)
    return array(
        "h",
        [
            int(32767 * amplitude * math.sin(2.0 * math.pi * 40.0 * (index / SAMPLE_RATE)))
            for index in range(total)
        ],
    )


def _write_wav(path: Path, *chunks: array) -> None:
    samples = array("h")
    for chunk in chunks:
        samples.extend(chunk)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


def _read_wav_samples(path: Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        samples = array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    return sample_rate, samples


def _peak(samples: array) -> int:
    return max((abs(sample) for sample in samples), default=0)


class StubVadBackend:
    name = "stub_vad"

    def __init__(self, evidence: BoundaryEvidence) -> None:
        self.evidence = evidence

    def analyze(self, *_args, **_kwargs) -> BoundaryEvidence:
        return self.evidence

    def available(self) -> bool:
        return True


class StubAsrBackend:
    name = "stub_asr"

    def __init__(self, transcript: TranscriptResult) -> None:
        self.transcript = transcript

    def transcribe(self, *_args, **_kwargs) -> TranscriptResult:
        return self.transcript

    def available(self) -> bool:
        return True


class PerFileAsrBackend:
    name = "per_file_stub_asr"

    def __init__(self, transcripts: dict[str, TranscriptResult]) -> None:
        self.transcripts = transcripts

    def transcribe(self, input_path: Path, *_args, **_kwargs) -> TranscriptResult:
        return self.transcripts[input_path.name]

    def available(self) -> bool:
        return True


class UnavailableVadBackend:
    name = "unavailable_vad"

    def available(self) -> bool:
        return False

    def check_ready(self) -> str | None:
        return "stub VAD unavailable"


class UnavailableAsrBackend:
    name = "unavailable_asr"

    def available(self) -> bool:
        return False

    def check_ready(self, _config=None) -> str | None:
        return "stub ASR unavailable"


def _default_preparer(tmp_path: Path, *, vad_backend=None, asr_backend=None) -> RawAudioPreparer:
    config = load_trim_config(None)
    config.paths.output_dir = str(tmp_path / "prepared")
    config.paths.artifact_dir = str(tmp_path / "artifacts")
    return RawAudioPreparer(config, vad_backend=vad_backend, asr_backend=asr_backend)


def _artifact_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def _output_dir(tmp_path: Path) -> Path:
    return tmp_path / "prepared"


def _build_transcript(*segments: TranscriptSegment) -> TranscriptResult:
    return TranscriptResult(
        text=" ".join(segment.text for segment in segments),
        segments=list(segments),
        backend="stub_asr",
        confidence=0.95,
        metadata={"available": True},
    )


def test_silence_only_file_is_flagged_for_review(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    _write_wav(audio_path, _silence(2.0))
    preparer = _default_preparer(tmp_path)

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-silence",
        dry_run=True,
    )

    assert result["manifest_row"]["manual_review"] is True
    assert "No speech window" in (result["decision"]["fallback_reason"] or "")
    assert result["manifest_row"]["final_start_offset_sec"] == pytest.approx(0.0)
    assert result["manifest_row"]["final_end_offset_sec"] == pytest.approx(2.0, abs=0.05)


def test_vad_can_override_short_noise_before_speech(tmp_path: Path) -> None:
    audio_path = tmp_path / "noise-before-speech.wav"
    _write_wav(
        audio_path,
        _silence(0.08),
        _noise(0.10, amplitude=0.05),
        _silence(0.27),
        _tone(1.0),
        _silence(0.25),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.33,
            end_offset_sec=1.65,
            speech_start_sec=0.45,
            speech_end_sec=1.45,
            confidence=0.92,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.45, end_sec=1.45, label="speech", confidence=0.92, source="stub_vad")],
        )
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend)

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad",
        dry_run=True,
    )

    assert result["manifest_row"]["method_chosen"] == "ffmpeg-vad"
    assert result["manifest_row"]["trim_mode"] == "onset_preservation"
    assert result["manifest_row"]["raw_opening_boundary_sec"] == pytest.approx(0.33, abs=0.05)
    assert result["manifest_row"]["final_start_offset_sec"] <= 0.15
    assert result["manifest_row"]["final_start_offset_sec"] < result["manifest_row"]["raw_opening_boundary_sec"]


def test_spoken_date_intro_is_removed_when_confident(tmp_path: Path) -> None:
    audio_path = tmp_path / "date-intro.wav"
    _write_wav(
        audio_path,
        _silence(0.50),
        _tone(2.50, frequency_hz=180.0),
        _silence(0.40),
        _tone(2.60, frequency_hz=260.0),
        _silence(0.30),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.38,
            end_offset_sec=6.15,
            speech_start_sec=0.50,
            speech_end_sec=5.95,
            confidence=0.93,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.50, end_sec=5.95, label="speech", confidence=0.93, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Today is Monday March 3rd 2025 journal entry",
            start_sec=0.50,
            end_sec=3.00,
            words=[
                TranscriptWord("Today", 0.50, 0.70),
                TranscriptWord("is", 0.70, 0.82),
                TranscriptWord("Monday", 0.82, 1.12),
                TranscriptWord("March", 1.12, 1.38),
                TranscriptWord("3rd", 1.38, 1.52),
                TranscriptWord("2025", 1.52, 1.74),
                TranscriptWord("journal", 1.74, 2.05),
                TranscriptWord("entry", 2.05, 2.28),
            ],
        ),
        TranscriptSegment(
            text="The actual reflection starts now",
            start_sec=3.40,
            end_sec=4.60,
            words=[
                TranscriptWord("The", 3.40, 3.50),
                TranscriptWord("actual", 3.50, 3.72),
                TranscriptWord("reflection", 3.72, 4.00),
                TranscriptWord("starts", 4.00, 4.18),
                TranscriptWord("now", 4.18, 4.30),
            ],
        ),
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=False,
    )

    assert result["manifest_row"]["method_chosen"] == "ffmpeg-vad-asr"
    assert result["manifest_row"]["spoken_intro_removed_sec"] >= 1.3
    assert result["manifest_row"]["opening_labels"] == "date|content"
    assert result["manifest_row"]["predicted_date_end_sec"] == pytest.approx(1.74, abs=0.05)
    assert result["intro_trim"]["min_overshoot_sec"] == pytest.approx(0.22, abs=0.001)
    assert result["intro_trim"]["max_overshoot_sec"] == pytest.approx(0.40, abs=0.001)
    assert result["intro_trim"]["overshoot_used_sec"] == pytest.approx(0.22, abs=0.001)
    assert result["intro_trim"]["initial_late_biased_trim_sec"] == pytest.approx(1.96, abs=0.05)
    assert result["intro_trim"]["snapped_trim_start_sec"] is None
    assert result["intro_trim"]["chosen_trim_start_sec"] == pytest.approx(1.96, abs=0.05)
    assert result["intro_trim"]["next_intro_boundary_sec"] == pytest.approx(1.74, abs=0.05)
    assert result["intro_trim"]["snap_found"] is False
    assert result["intro_trim"]["fade_in_applied_ms"] == 15
    assert result["decision"]["manual_review"] is True
    assert "extremely close" in " ".join(result["decision"]["manual_review_reasons"])
    assert Path(result["output"]["path"]).exists()


def test_content_without_date_intro_is_left_intact(tmp_path: Path) -> None:
    audio_path = tmp_path / "no-date-intro.wav"
    _write_wav(
        audio_path,
        _silence(0.40),
        _tone(2.20, frequency_hz=210.0),
        _silence(0.25),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.28,
            end_offset_sec=2.85,
            speech_start_sec=0.40,
            speech_end_sec=2.65,
            confidence=0.91,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.40, end_sec=2.65, label="speech", confidence=0.91, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Welcome back to the meditation",
            start_sec=0.40,
            end_sec=2.30,
            words=[
                TranscriptWord("Welcome", 0.40, 0.58),
                TranscriptWord("back", 0.58, 0.74),
                TranscriptWord("to", 0.74, 0.82),
                TranscriptWord("the", 0.82, 0.92),
                TranscriptWord("meditation", 0.92, 1.20),
            ],
        )
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=True,
    )

    assert result["manifest_row"]["spoken_intro_removed_sec"] == pytest.approx(0.0)
    assert result["manifest_row"]["trim_mode"] == "onset_preservation"
    assert result["manifest_row"]["raw_opening_boundary_sec"] == pytest.approx(0.28, abs=0.05)
    assert result["manifest_row"]["opening_preroll_applied_sec"] >= 0.15
    assert result["manifest_row"]["final_start_offset_sec"] <= 0.10
    assert result["decision"]["manual_review"] is False
    assert result["intro_trim"]["detected"] is False
    assert result["manifest_row"]["fade_in_applied_ms"] == 15


def test_fast_spoken_date_near_content_boundary_triggers_manual_review(tmp_path: Path) -> None:
    audio_path = tmp_path / "fast-date-content.wav"
    _write_wav(
        audio_path,
        _silence(0.30),
        _tone(0.85, frequency_hz=185.0),
        _silence(0.05),
        _tone(1.20, frequency_hz=255.0),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.20,
            end_offset_sec=2.30,
            speech_start_sec=0.30,
            speech_end_sec=2.20,
            confidence=0.94,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.30, end_sec=2.20, label="speech", confidence=0.94, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Today is March 3rd 2025",
            start_sec=0.30,
            end_sec=0.92,
            words=[
                TranscriptWord("Today", 0.30, 0.42),
                TranscriptWord("is", 0.42, 0.48),
                TranscriptWord("March", 0.48, 0.68),
                TranscriptWord("3rd", 0.68, 0.78),
                TranscriptWord("2025", 0.78, 0.92),
            ],
        ),
        TranscriptSegment(
            text="Breathing starts now",
            start_sec=0.96,
            end_sec=1.45,
            words=[
                TranscriptWord("Breathing", 0.96, 1.14),
                TranscriptWord("starts", 1.14, 1.28),
                TranscriptWord("now", 1.28, 1.38),
            ],
        ),
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=True,
    )

    assert result["manifest_row"]["opening_labels"] == "date|content"
    assert result["intro_trim"]["min_overshoot_sec"] == pytest.approx(0.22, abs=0.001)
    assert result["intro_trim"]["initial_late_biased_trim_sec"] == pytest.approx(1.14, abs=0.05)
    assert result["intro_trim"]["next_intro_boundary_sec"] == pytest.approx(0.96, abs=0.05)
    assert result["intro_trim"]["snap_found"] is True
    assert result["intro_trim"]["snapped_trim_start_sec"] == pytest.approx(1.17, abs=0.06)
    assert result["intro_trim"]["chosen_trim_start_sec"] >= result["intro_trim"]["initial_late_biased_trim_sec"]
    assert result["intro_trim"]["overshoot_used_sec"] > result["intro_trim"]["min_overshoot_sec"]
    assert result["decision"]["manual_review"] is True
    assert "Spoken date ended extremely close" in " ".join(result["decision"]["manual_review_reasons"])


def test_date_followed_by_title_hint_is_removed_before_content(tmp_path: Path) -> None:
    audio_path = tmp_path / "dp_2026-03-07_inspiration_ricardo-audio.wav"
    _write_wav(
        audio_path,
        _silence(0.40),
        _tone(1.20, frequency_hz=175.0),
        _silence(0.20),
        _tone(0.70, frequency_hz=190.0),
        _silence(0.20),
        _tone(1.60, frequency_hz=250.0),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.28,
            end_offset_sec=4.40,
            speech_start_sec=0.40,
            speech_end_sec=4.20,
            confidence=0.92,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.40, end_sec=4.20, label="speech", confidence=0.92, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="7 of March",
            start_sec=0.40,
            end_sec=1.20,
            words=[
                TranscriptWord("7", 0.40, 0.50),
                TranscriptWord("of", 0.50, 0.58),
                TranscriptWord("March", 0.58, 0.84),
            ],
        ),
        TranscriptSegment(
            text="Inspiration",
            start_sec=1.40,
            end_sec=1.95,
            words=[
                TranscriptWord("Inspiration", 1.40, 1.90),
            ],
        ),
        TranscriptSegment(
            text="The intimacy of the relationship opens our prayer",
            start_sec=2.20,
            end_sec=3.90,
            words=[
                TranscriptWord("The", 2.20, 2.30),
                TranscriptWord("intimacy", 2.30, 2.58),
                TranscriptWord("of", 2.58, 2.66),
                TranscriptWord("the", 2.66, 2.76),
                TranscriptWord("relationship", 2.76, 3.10),
                TranscriptWord("opens", 3.10, 3.30),
                TranscriptWord("our", 3.30, 3.40),
                TranscriptWord("prayer", 3.40, 3.62),
            ],
        ),
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=True,
    )

    assert result["manifest_row"]["method_chosen"] == "ffmpeg-vad-asr"
    assert result["manifest_row"]["spoken_intro_removed_sec"] < 1.0
    assert result["manifest_row"]["predicted_date_end_sec"] == pytest.approx(0.84, abs=0.05)
    assert result["manifest_row"]["predicted_title_start_sec"] == pytest.approx(1.40, abs=0.05)
    assert result["manifest_row"]["predicted_content_start_sec"] == pytest.approx(2.20, abs=0.05)
    assert result["manifest_row"]["opening_labels"] == "date|title|content"
    assert result["manifest_row"]["final_start_offset_sec"] < 1.40
    assert result["intro_detection"]["detected"] is True
    assert "context:inspiration" in result["intro_detection"]["matched_patterns"]
    assert result["intro_trim"]["next_preserved_label"] == "title"
    assert result["intro_trim"]["overshoot_used_sec"] == pytest.approx(0.22, abs=0.001)
    assert result["intro_trim"]["snap_found"] is False
    assert result["decision"]["manual_review"] is False


def test_title_without_date_is_preserved_even_when_it_matches_context(tmp_path: Path) -> None:
    audio_path = tmp_path / "dp_2026-03-07_inspiration_ricardo-audio.wav"
    _write_wav(
        audio_path,
        _silence(0.30),
        _tone(0.70, frequency_hz=190.0),
        _silence(0.18),
        _tone(1.50, frequency_hz=250.0),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.18,
            end_offset_sec=2.78,
            speech_start_sec=0.30,
            speech_end_sec=2.68,
            confidence=0.92,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.30, end_sec=2.68, label="speech", confidence=0.92, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Inspiration",
            start_sec=0.30,
            end_sec=0.82,
            words=[
                TranscriptWord("Inspiration", 0.30, 0.82),
            ],
        ),
        TranscriptSegment(
            text="The intimacy of the relationship opens our prayer",
            start_sec=1.00,
            end_sec=2.40,
            words=[
                TranscriptWord("The", 1.00, 1.10),
                TranscriptWord("intimacy", 1.10, 1.38),
                TranscriptWord("of", 1.38, 1.46),
                TranscriptWord("the", 1.46, 1.56),
                TranscriptWord("relationship", 1.56, 1.90),
                TranscriptWord("opens", 1.90, 2.10),
                TranscriptWord("our", 2.10, 2.20),
                TranscriptWord("prayer", 2.20, 2.42),
            ],
        ),
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=True,
    )

    assert result["intro_detection"]["detected"] is False
    assert result["manifest_row"]["spoken_intro_removed_sec"] == pytest.approx(0.0)
    assert result["manifest_row"]["opening_labels"] == "title|content"
    assert result["manifest_row"]["predicted_title_start_sec"] == pytest.approx(0.30, abs=0.05)
    assert result["manifest_row"]["trim_mode"] == "onset_preservation"
    assert result["manifest_row"]["final_start_offset_sec"] <= 0.02
    assert result["manifest_row"]["final_start_offset_sec"] < result["manifest_row"]["raw_opening_boundary_sec"]


def test_attachment_metadata_can_restore_context_phrases_after_task_id_download(tmp_path: Path) -> None:
    audio_path = tmp_path / "86abc123.wav"
    meta_dir = tmp_path / ".metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "86abc123.json").write_text(
        json.dumps(
            {
                "task_id": "86abc123",
                "task_name": "Gospel",
                "selected_attachment": {
                    "title": "dp_2026-03-29_gospel_katherine-audio(1).m4a",
                    "url_filename": "dp_2026-03-29_gospel_katherine-audio(1).m4a",
                },
            }
        ),
        encoding="utf-8",
    )
    _write_wav(
        audio_path,
        _silence(0.35),
        _tone(0.90, frequency_hz=175.0),
        _silence(0.18),
        _tone(0.70, frequency_hz=190.0),
        _silence(0.18),
        _tone(1.30, frequency_hz=250.0),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.23,
            end_offset_sec=3.70,
            speech_start_sec=0.35,
            speech_end_sec=3.58,
            confidence=0.92,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.35, end_sec=3.58, label="speech", confidence=0.92, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="29 of March",
            start_sec=0.35,
            end_sec=0.95,
            words=[
                TranscriptWord("29", 0.35, 0.45),
                TranscriptWord("of", 0.45, 0.52),
                TranscriptWord("March", 0.52, 0.74),
            ],
        ),
        TranscriptSegment(
            text="Gospel",
            start_sec=1.10,
            end_sec=1.52,
            words=[
                TranscriptWord("Gospel", 1.10, 1.50),
            ],
        ),
        TranscriptSegment(
            text="The reflection begins here",
            start_sec=1.80,
            end_sec=2.90,
            words=[
                TranscriptWord("The", 1.80, 1.88),
                TranscriptWord("reflection", 1.88, 2.15),
                TranscriptWord("begins", 2.15, 2.35),
                TranscriptWord("here", 2.35, 2.48),
            ],
        ),
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=True,
    )

    assert result["manifest_row"]["opening_labels"] == "date|title|content"
    assert "context:gospel" in result["intro_detection"]["matched_patterns"]
    assert "gospel" in [phrase.lower() for phrase in result["context_phrases"]]


def test_date_with_missing_following_boundary_is_flagged_for_review(tmp_path: Path) -> None:
    audio_path = tmp_path / "date-only-window.wav"
    _write_wav(
        audio_path,
        _silence(0.25),
        _tone(1.10, frequency_hz=175.0),
        _silence(0.20),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.15,
            end_offset_sec=1.65,
            speech_start_sec=0.25,
            speech_end_sec=1.35,
            confidence=0.92,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.25, end_sec=1.35, label="speech", confidence=0.92, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Recorded on March 7th 2025",
            start_sec=0.25,
            end_sec=0.82,
            words=[
                TranscriptWord("Recorded", 0.25, 0.36),
                TranscriptWord("on", 0.36, 0.44),
                TranscriptWord("March", 0.44, 0.62),
                TranscriptWord("7th", 0.62, 0.72),
                TranscriptWord("2025", 0.72, 0.82),
            ],
        )
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=True,
    )

    assert result["intro_detection"]["detected"] is True
    assert result["intro_trim"]["initial_late_biased_trim_sec"] == pytest.approx(1.04, abs=0.05)
    assert result["intro_trim"]["chosen_trim_start_sec"] == pytest.approx(1.04, abs=0.05)
    assert result["intro_trim"]["next_intro_boundary_sec"] is None
    assert result["decision"]["manual_review"] is True
    assert "no reliable next word/content boundary" in " ".join(result["decision"]["manual_review_reasons"]).lower()
    assert "segmentation was missing" in " ".join(result["decision"]["manual_review_reasons"]).lower()


def test_date_trim_snaps_to_cleaner_boundary_and_fades_in_output(tmp_path: Path) -> None:
    audio_path = tmp_path / "date-tail-cleanup.wav"
    _write_wav(
        audio_path,
        _silence(0.30),
        _tone(0.60, frequency_hz=185.0, amplitude=0.35),
        _decay_tone(0.20, frequency_hz=185.0, start_amplitude=0.10, end_amplitude=0.0),
        _silence(0.06),
        _tone(0.90, frequency_hz=255.0, amplitude=0.30),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.20,
            end_offset_sec=2.10,
            speech_start_sec=0.30,
            speech_end_sec=2.06,
            confidence=0.94,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.30, end_sec=2.06, label="speech", confidence=0.94, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Today is March 3rd 2025",
            start_sec=0.30,
            end_sec=0.90,
            words=[
                TranscriptWord("Today", 0.30, 0.42),
                TranscriptWord("is", 0.42, 0.48),
                TranscriptWord("March", 0.48, 0.68),
                TranscriptWord("3rd", 0.68, 0.78),
                TranscriptWord("2025", 0.78, 0.90),
            ],
        ),
        TranscriptSegment(
            text="The reflection begins now",
            start_sec=1.16,
            end_sec=1.80,
            words=[
                TranscriptWord("The", 1.16, 1.24),
                TranscriptWord("reflection", 1.24, 1.50),
                TranscriptWord("begins", 1.50, 1.68),
                TranscriptWord("now", 1.68, 1.78),
            ],
        ),
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=False,
    )

    output_path = Path(result["output"]["path"])
    assert output_path.exists()
    assert result["intro_trim"]["initial_late_biased_trim_sec"] == pytest.approx(1.12, abs=0.05)
    assert result["intro_trim"]["snap_found"] is True
    assert result["intro_trim"]["snapped_trim_start_sec"] == pytest.approx(1.12, abs=0.05)
    assert result["intro_trim"]["chosen_trim_start_sec"] == result["intro_trim"]["snapped_trim_start_sec"]
    assert result["output"]["leading_fade_in_ms"] == 15

    sample_rate, samples = _read_wav_samples(output_path)
    first_5ms = samples[: int(sample_rate * 0.005)]
    later_20ms = samples[int(sample_rate * 0.090) : int(sample_rate * 0.100)]
    assert _peak(first_5ms) < 500
    assert _peak(later_20ms) > 1500


def test_no_date_intro_preserves_inhale_and_first_word_onset(tmp_path: Path) -> None:
    audio_path = tmp_path / "no-date-inhale.wav"
    _write_wav(
        audio_path,
        _silence(0.20),
        _noise(0.05, amplitude=0.01),
        _tone(0.80, frequency_hz=230.0, amplitude=0.28),
        _silence(0.20),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.33,
            end_offset_sec=1.30,
            speech_start_sec=0.25,
            speech_end_sec=1.05,
            confidence=0.93,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.25, end_sec=1.05, label="speech", confidence=0.93, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Inspiration begins here",
            start_sec=0.25,
            end_sec=0.88,
            words=[
                TranscriptWord("Inspiration", 0.25, 0.52),
                TranscriptWord("begins", 0.52, 0.72),
                TranscriptWord("here", 0.72, 0.82),
            ],
        )
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=False,
    )

    assert result["manifest_row"]["trim_mode"] == "onset_preservation"
    assert result["manifest_row"]["date_detected"] is False
    assert result["manifest_row"]["raw_opening_boundary_sec"] == pytest.approx(0.33, abs=0.05)
    assert result["manifest_row"]["final_start_offset_sec"] < 0.25
    assert result["manifest_row"]["final_start_offset_sec"] >= 0.05
    assert result["manifest_row"]["opening_preroll_applied_sec"] >= 0.15
    assert result["intro_trim"]["snap_found"] is True
    assert result["output"]["leading_fade_in_ms"] == 15

    sample_rate, samples = _read_wav_samples(Path(result["output"]["path"]))
    # With larger preroll the trim starts earlier in silence; the inhale
    # (noise) sits at ~70-120 ms into the output and tone follows after.
    inhale_region = samples[int(sample_rate * 0.060) : int(sample_rate * 0.120)]
    tone_region = samples[int(sample_rate * 0.130) : int(sample_rate * 0.160)]
    assert _peak(inhale_region) > 20, "Inhale was clipped from output"
    assert _peak(tone_region) > 1500, "Content tone missing from output"


def test_prepare_file_writes_prepared_metadata_with_task_mapping(tmp_path: Path) -> None:
    audio_path = tmp_path / "dp_2026-04-29_inspiration_dinah-audio.wav"
    meta_dir = tmp_path / ".metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "dp_2026-04-29_inspiration_dinah-audio.json").write_text(
        json.dumps(
            {
                "task_id": "869cjgxgj",
                "task_name": "dp_2026-04-29_inspiration_dinah-audio",
                "download_filename": "dp_2026-04-29_inspiration_dinah-audio.wav",
            }
        ),
        encoding="utf-8",
    )
    _write_wav(audio_path, _silence(0.30), _tone(1.0), _silence(0.20))
    preparer = _default_preparer(tmp_path)

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-silence",
        dry_run=False,
    )

    prepared_meta_path = _output_dir(tmp_path) / ".metadata" / "dp_2026-04-29_inspiration_dinah-audio.json"
    prepared_metadata = json.loads(prepared_meta_path.read_text(encoding="utf-8"))

    assert Path(result["output"]["path"]).exists()
    assert prepared_metadata["task_id"] == "869cjgxgj"
    assert prepared_metadata["prepared_filename"] == "dp_2026-04-29_inspiration_dinah-audio.wav"
    assert prepared_metadata["source_input_filename"] == "dp_2026-04-29_inspiration_dinah-audio.wav"


def test_real_words_close_to_the_beginning_are_not_clipped(tmp_path: Path) -> None:
    audio_path = tmp_path / "near-zero-speech.wav"
    _write_wav(
        audio_path,
        _silence(0.04),
        _tone(1.20, frequency_hz=240.0),
        _silence(0.20),
    )
    preparer = _default_preparer(tmp_path)

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-silence",
        dry_run=True,
    )

    assert result["manifest_row"]["final_start_offset_sec"] <= 0.02


def test_requested_vad_method_fails_fast_when_vad_is_unavailable(tmp_path: Path) -> None:
    audio_path = tmp_path / "need-vad.wav"
    _write_wav(audio_path, _silence(0.20), _tone(1.0))
    preparer = _default_preparer(tmp_path, vad_backend=UnavailableVadBackend())

    with pytest.raises(RuntimeError, match="ffmpeg-vad"):
        preparer.prepare_file(
            audio_path,
            output_dir=_output_dir(tmp_path),
            artifact_root=_artifact_dir(tmp_path),
            requested_method="ffmpeg-vad",
            dry_run=True,
        )


def test_requested_asr_method_fails_fast_when_asr_is_unavailable(tmp_path: Path) -> None:
    audio_path = tmp_path / "need-asr.wav"
    _write_wav(audio_path, _silence(0.20), _tone(1.0))
    preparer = _default_preparer(
        tmp_path,
        vad_backend=StubVadBackend(
            BoundaryEvidence(
                backend="stub_vad",
                available=True,
                start_offset_sec=0.1,
                end_offset_sec=1.1,
                speech_start_sec=0.2,
                speech_end_sec=1.0,
                confidence=0.9,
                reason="Synthetic VAD evidence.",
            )
        ),
        asr_backend=UnavailableAsrBackend(),
    )

    with pytest.raises(RuntimeError, match="ffmpeg-vad-asr"):
        preparer.prepare_file(
            audio_path,
            output_dir=_output_dir(tmp_path),
            artifact_root=_artifact_dir(tmp_path),
            requested_method="ffmpeg-vad-asr",
            dry_run=True,
        )


def test_invalid_intro_classification_raises_during_single_file_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio_path = tmp_path / "bad-intro.wav"
    _write_wav(audio_path, _silence(0.20), _tone(1.0))
    preparer = _default_preparer(
        tmp_path,
        vad_backend=StubVadBackend(
            BoundaryEvidence(
                backend="stub_vad",
                available=True,
                start_offset_sec=0.1,
                end_offset_sec=1.1,
                speech_start_sec=0.2,
                speech_end_sec=1.0,
                confidence=0.9,
                reason="Synthetic VAD evidence.",
            )
        ),
        asr_backend=StubAsrBackend(
            _build_transcript(
                TranscriptSegment(
                    text="March 7",
                    start_sec=0.2,
                    end_sec=0.6,
                    words=[
                        TranscriptWord("March", 0.2, 0.4),
                        TranscriptWord("7", 0.4, 0.5),
                    ],
                )
            )
        ),
    )

    monkeypatch.setattr(
        pipeline_module,
        "detect_intro",
        lambda *_args, **_kwargs: IntroDetectionResult(
            detected=True,
            trim_offset_sec=0.6,
            confidence=0.9,
            classified_segments=[
                IntroSegmentClassification(
                    label="title",
                    start_sec=0.2,
                    end_sec=0.6,
                    text="March 7",
                    confidence=0.9,
                    removable=True,
                )
            ],
        ),
    )

    with pytest.raises(pipeline_module.IntroDetectionValidationError, match="removable"):
        preparer.prepare_file(
            audio_path,
            output_dir=_output_dir(tmp_path),
            artifact_root=_artifact_dir(tmp_path),
            requested_method="ffmpeg-vad-asr",
            dry_run=True,
        )


def test_prepare_batch_continues_past_file_level_intro_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good_audio = tmp_path / "good.wav"
    bad_audio = tmp_path / "bad.wav"
    _write_wav(good_audio, _silence(0.25), _tone(1.0), _silence(0.15))
    _write_wav(bad_audio, _silence(0.25), _tone(1.0), _silence(0.15))

    output_dir = _output_dir(tmp_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_output = output_dir / "bad.wav"
    stale_output.write_text("stale prepared output", encoding="utf-8")

    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.13,
            end_offset_sec=1.32,
            speech_start_sec=0.25,
            speech_end_sec=1.15,
            confidence=0.92,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.25, end_sec=1.15, label="speech", confidence=0.92, source="stub_vad")],
        )
    )
    preparer = _default_preparer(
        tmp_path,
        vad_backend=vad_backend,
        asr_backend=PerFileAsrBackend(
            {
                "good.wav": _build_transcript(
                    TranscriptSegment(
                        text="Welcome to the reflection",
                        start_sec=0.25,
                        end_sec=1.0,
                        words=[
                            TranscriptWord("Welcome", 0.25, 0.45),
                            TranscriptWord("to", 0.45, 0.52),
                            TranscriptWord("the", 0.52, 0.60),
                            TranscriptWord("reflection", 0.60, 0.92),
                        ],
                    )
                ),
                "bad.wav": _build_transcript(
                    TranscriptSegment(
                        text="March 7",
                        start_sec=0.25,
                        end_sec=0.60,
                        words=[
                            TranscriptWord("March", 0.25, 0.45),
                            TranscriptWord("7", 0.45, 0.55),
                        ],
                    )
                ),
            }
        ),
    )

    original_detect_intro = pipeline_module.detect_intro

    def _fake_detect_intro(transcript: TranscriptResult, *args, **kwargs) -> IntroDetectionResult:
        if transcript.text == "March 7":
            return IntroDetectionResult(
                detected=True,
                trim_offset_sec=0.60,
                confidence=0.9,
                classified_segments=[
                    IntroSegmentClassification(
                        label="title",
                        start_sec=0.25,
                        end_sec=0.60,
                        text="March 7",
                        confidence=0.9,
                        removable=True,
                    )
                ],
            )
        return original_detect_intro(transcript, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "detect_intro", _fake_detect_intro)

    result = preparer.prepare_batch(
        input_dir=tmp_path,
        output_dir=output_dir,
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        file_names=["good.wav", "bad.wav"],
        dry_run=False,
    )

    assert result["summary"]["files_seen"] == 2
    assert result["summary"]["files_emitted"] == 1
    assert result["summary"]["files_failed"] == 1

    rows_by_id = {row["file_id"]: row for row in result["rows"]}
    assert rows_by_id["good"]["method_chosen"] == "ffmpeg-vad-asr"
    assert rows_by_id["bad"]["method_chosen"] == "error"
    assert "IntroDetectionValidationError" in rows_by_id["bad"]["failure_or_fallback_reason"]
    assert rows_by_id["bad"]["manual_review"] is True
    assert stale_output.exists() is False

    bad_sidecar = _artifact_dir(tmp_path) / "files" / "bad" / "bad.trim.json"
    payload = json.loads(bad_sidecar.read_text(encoding="utf-8"))
    assert payload["output"]["stale_output_removed"] is True
    assert payload["error"]["type"] == "IntroDetectionValidationError"


def test_forward_snap_picks_deepest_dip_not_first_below_threshold(tmp_path: Path) -> None:
    """The forward snap should land at the deepest energy dip in the window,
    not the first frame that happens to cross the threshold.  This prevents
    residual consonant tails from surviving when the first quiet frame is
    only marginally below threshold while a much deeper dip exists later."""
    from trim_raw_audio.audio import find_opening_boundary_snap

    audio_path = tmp_path / "two-dips.wav"
    # Layout: marginal dip (-36 dB zone) then deep silence then tone
    _write_wav(
        audio_path,
        _noise(0.03, amplitude=0.012),       # ~-38 dB, marginal
        _silence(0.04),                       # deep dip (~-inf dB)
        _tone(0.08, frequency_hz=300.0),      # loud content
    )
    snap = find_opening_boundary_snap(
        audio_path,
        search_start_sec=0.0,
        search_end_sec=0.10,
        frame_ms=5,
        threshold_db=-35.0,
        direction="forward",
    )

    assert snap["found"] is True
    # Should land in the silence region (0.03-0.07), NOT in the marginal noise (0.0-0.03)
    assert snap["snapped_trim_start_sec"] >= 0.025
    assert snap["snapped_trim_start_sec"] <= 0.07
    assert "deepest" in snap["reason"]


def test_date_tail_residual_is_cleared_by_deepest_dip(tmp_path: Path) -> None:
    """End-to-end: a spoken date ending with a decaying consonant tail should
    be fully cleared.  The trim should land in the silence gap, not in the
    tail, producing an output whose first 20 ms has negligible energy."""
    audio_path = tmp_path / "date-consonant-tail.wav"
    _write_wav(
        audio_path,
        _silence(0.25),
        _tone(0.55, frequency_hz=180.0, amplitude=0.35),       # "March 3rd 2025"
        _decay_tone(0.18, frequency_hz=180.0, start_amplitude=0.08, end_amplitude=0.0),
        _silence(0.10),                                         # clean gap
        _tone(1.20, frequency_hz=250.0, amplitude=0.30),       # content
        _silence(0.20),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.15,
            end_offset_sec=2.38,
            speech_start_sec=0.25,
            speech_end_sec=2.28,
            confidence=0.93,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.25, end_sec=2.28, label="speech", confidence=0.93, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Today is March 3rd 2025",
            start_sec=0.25,
            end_sec=0.80,
            words=[
                TranscriptWord("Today", 0.25, 0.36),
                TranscriptWord("is", 0.36, 0.42),
                TranscriptWord("March", 0.42, 0.58),
                TranscriptWord("3rd", 0.58, 0.68),
                TranscriptWord("2025", 0.68, 0.80),
            ],
        ),
        TranscriptSegment(
            text="The reflection opens now",
            start_sec=1.08,
            end_sec=1.90,
            words=[
                TranscriptWord("The", 1.08, 1.16),
                TranscriptWord("reflection", 1.16, 1.42),
                TranscriptWord("opens", 1.42, 1.60),
                TranscriptWord("now", 1.60, 1.72),
            ],
        ),
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=False,
    )

    assert result["intro_trim"]["trim_mode"] == "date_removal"
    assert result["intro_trim"]["snap_found"] is True
    # Snap should land in the silence gap (0.98-1.08), well past the decay tail
    chosen = result["intro_trim"]["chosen_trim_start_sec"]
    assert chosen >= 0.95, f"Trim landed at {chosen}, likely inside the consonant tail"
    assert chosen <= 1.08, f"Trim landed at {chosen}, past the silence gap into content"
    assert result["output"]["leading_fade_in_ms"] == 15

    # Verify output audio starts clean
    output_path = Path(result["output"]["path"])
    assert output_path.exists()
    sample_rate, samples = _read_wav_samples(output_path)
    first_20ms = samples[: int(sample_rate * 0.020)]
    assert _peak(first_20ms) < 600, f"Residual energy in first 20ms: peak={_peak(first_20ms)}"


def test_onset_preservation_protects_full_inhale(tmp_path: Path) -> None:
    """When no date is detected, the trim should pull back far enough
    to include a preparatory inhale before the first spoken word,
    even if VAD reports speech starting after the inhale."""
    audio_path = tmp_path / "inhale-then-speech.wav"
    _write_wav(
        audio_path,
        _silence(0.15),
        _noise(0.18, amplitude=0.015),                          # inhale
        _tone(1.00, frequency_hz=230.0, amplitude=0.30),        # "Inspiration..."
        _silence(0.20),
    )
    vad_backend = StubVadBackend(
        BoundaryEvidence(
            backend="stub_vad",
            available=True,
            start_offset_sec=0.28,
            end_offset_sec=1.48,
            speech_start_sec=0.33,
            speech_end_sec=1.33,
            confidence=0.92,
            reason="Synthetic VAD evidence.",
            spans=[Span(start_sec=0.33, end_sec=1.33, label="speech", confidence=0.92, source="stub_vad")],
        )
    )
    transcript = _build_transcript(
        TranscriptSegment(
            text="Inspiration opens everything",
            start_sec=0.33,
            end_sec=1.15,
            words=[
                TranscriptWord("Inspiration", 0.33, 0.60),
                TranscriptWord("opens", 0.60, 0.80),
                TranscriptWord("everything", 0.80, 1.05),
            ],
        )
    )
    preparer = _default_preparer(tmp_path, vad_backend=vad_backend, asr_backend=StubAsrBackend(transcript))

    result = preparer.prepare_file(
        audio_path,
        output_dir=_output_dir(tmp_path),
        artifact_root=_artifact_dir(tmp_path),
        requested_method="ffmpeg-vad-asr",
        dry_run=False,
    )

    assert result["manifest_row"]["trim_mode"] == "onset_preservation"
    final_start = result["manifest_row"]["final_start_offset_sec"]
    # Trim should start before the inhale begins (0.15), or at least at the
    # very start of the inhale, preserving the natural breath
    assert final_start <= 0.18, f"Trim at {final_start} clips into the inhale"
    assert result["manifest_row"]["opening_preroll_applied_sec"] >= 0.15

    # Verify the first spoken word is fully intact in the output
    output_path = Path(result["output"]["path"])
    assert output_path.exists()
    sample_rate, samples = _read_wav_samples(output_path)
    # The inhale should be present in the first ~200ms of output
    inhale_region = samples[: int(sample_rate * 0.20)]
    assert _peak(inhale_region) > 30, "Inhale was clipped from output"
