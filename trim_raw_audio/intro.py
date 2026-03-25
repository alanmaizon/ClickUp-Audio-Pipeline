from __future__ import annotations

import re
from typing import Iterable

from .config import IntroClassifierConfig
from .models import IntroDetectionResult, IntroSegmentClassification, TranscriptResult, TranscriptSegment, TranscriptWord


_MULTISPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
)
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ORDINAL_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\b")
_DATE_PHRASE_RE = re.compile(r"\b(today is|it is|this is|recorded on|on this)\b")
_HEADER_RE = re.compile(
    r"\b(recording|session|entry|journal|memo|log|note|voice note|check in|check-in)\b"
)
_OPENING_DATE_REGEXES = [
    re.compile(
        rf"^(?:(?:today is|it is|this is|recorded on|on this)\s+)?"
        rf"(?:(?:{_WEEKDAY_RE.pattern[2:-2]})\s+)?"
        rf"(?:the\s+)?\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+(?:{_MONTH_RE.pattern[2:-2]})(?:\s+(?:19|20)\d{{2}})?\b"
    ),
    re.compile(
        rf"^(?:(?:today is|it is|this is|recorded on|on this)\s+)?"
        rf"(?:(?:{_WEEKDAY_RE.pattern[2:-2]})\s+)?"
        rf"(?:{_MONTH_RE.pattern[2:-2]})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s+(?:19|20)\d{{2}})?\b"
    ),
    re.compile(
        rf"^(?:(?:today is|it is|this is|recorded on|on this)\s+)?"
        rf"(?:{_WEEKDAY_RE.pattern[2:-2]})\s+(?:{_MONTH_RE.pattern[2:-2]})(?:\s+\d{{1,2}}(?:st|nd|rd|th)?)?(?:\s+(?:19|20)\d{{2}})?\b"
    ),
]
_DAY_OF_MONTH_RE = re.compile(
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{_MONTH_RE.pattern[2:-2]})\b"
)


def normalize_text(text: str) -> str:
    normalized = text.lower().replace("’", "'")
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    return _MULTISPACE_RE.sub(" ", normalized).strip()


def _collect_words(transcript: TranscriptResult) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for segment in transcript.segments:
        words.extend(segment.words)
    return words


def _word_timeline(words: Iterable[TranscriptWord]) -> tuple[str, list[tuple[int, int, TranscriptWord]]]:
    pieces: list[str] = []
    index: list[tuple[int, int, TranscriptWord]] = []
    cursor = 0
    for word in words:
        normalized = normalize_text(word.word)
        if not normalized:
            continue
        if pieces:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(normalized)
        cursor += len(normalized)
        index.append((start, cursor, word))
    return "".join(pieces), index


def _match_end_time_from_words(
    regexes: list[re.Pattern[str]],
    words: list[TranscriptWord],
    *,
    max_start_chars: int = 8,
) -> float | None:
    text, index = _word_timeline(words)
    if not text or not index:
        return None

    candidate_end: float | None = None
    for regex in regexes:
        for match in regex.finditer(text):
            if match.start() > max_start_chars:
                continue
            match_end = match.end()
            for start, end, word in index:
                if end >= match_end:
                    candidate_end = max(candidate_end or 0.0, word.end_sec)
                    break
    return candidate_end


def _first_context_match(text: str, context_phrases: list[str]) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None

    for phrase in context_phrases:
        phrase_normalized = normalize_text(phrase)
        if not phrase_normalized:
            continue
        if normalized == phrase_normalized or normalized in phrase_normalized or phrase_normalized in normalized:
            return phrase
    return None


def _score_date_text(
    text: str,
    repeated_phrases: list[str],
    boilerplate_phrases: list[str],
) -> tuple[float, list[str]]:
    score = 0.0
    matches: list[str] = []
    normalized = normalize_text(text)
    if not normalized:
        return 0.0, matches

    has_weekday = bool(_WEEKDAY_RE.search(normalized))
    has_month = bool(_MONTH_RE.search(normalized))
    has_year = bool(_YEAR_RE.search(normalized))
    has_ordinal = bool(_ORDINAL_RE.search(normalized))
    has_date_phrase = bool(_DATE_PHRASE_RE.search(normalized))
    has_header = bool(_HEADER_RE.search(normalized))
    has_day_of_month = bool(_DAY_OF_MONTH_RE.search(normalized))

    explicit_date = has_weekday or has_month or has_year
    date_combo = (has_month and (has_year or has_ordinal)) or (has_weekday and (has_month or has_year))

    repeated_match = next(
        (phrase for phrase in repeated_phrases if normalized.startswith(normalize_text(phrase))),
        None,
    )
    boilerplate_match = next(
        (phrase for phrase in boilerplate_phrases if normalize_text(phrase) in normalized),
        None,
    )

    if repeated_match:
        score += 0.08
        matches.append(f"repeated:{repeated_match}")
    if boilerplate_match:
        score += 0.10
        matches.append(f"boilerplate:{boilerplate_match}")
    if has_date_phrase:
        score += 0.20
        matches.append("date_phrase")
    if has_weekday:
        score += 0.12
        matches.append("weekday")
    if has_month:
        score += 0.26
        matches.append("month")
    if has_year:
        score += 0.20
        matches.append("year")
    if has_ordinal:
        score += 0.08
        matches.append("day_number")
    if has_day_of_month:
        score += 0.14
        matches.append("day_of_month")
    if date_combo:
        score += 0.32
        matches.append("date_combo")
    elif explicit_date and (has_year or has_ordinal or has_date_phrase):
        score += 0.18
        matches.append("explicit_date")
    if has_header and (has_date_phrase or explicit_date):
        score += 0.08
        matches.append("header_keyword")

    strong_enough = bool(
        repeated_match
        or boilerplate_match
        or date_combo
        or (explicit_date and has_date_phrase)
        or (has_month and has_ordinal)
    )
    if not strong_enough and not (has_month and has_ordinal):
        score = min(score, 0.44)
    return min(score, 0.97), matches


def _score_title_segment(
    segment: TranscriptSegment,
    *,
    context_phrases: list[str],
    config: IntroClassifierConfig,
    repeated_phrases: list[str],
    boilerplate_phrases: list[str],
    after_date: bool,
) -> tuple[float, list[str]]:
    normalized = normalize_text(segment.text)
    if not normalized:
        return 0.0, []

    date_score, _ = _score_date_text(segment.text, repeated_phrases, boilerplate_phrases)
    if date_score >= config.date_segment_min_score:
        return 0.0, []

    word_count = len(normalized.split())
    duration_sec = max(0.0, segment.end_sec - segment.start_sec)
    context_match = _first_context_match(segment.text, context_phrases)
    matches: list[str] = []
    score = 0.0

    if context_match:
        score += 0.72
        matches.append(f"context:{context_match}")
    if word_count <= config.title_max_words:
        score += 0.12
        matches.append("short_phrase")
    if word_count <= 3:
        score += 0.08
        matches.append("very_short_phrase")
    if duration_sec <= config.title_max_duration_sec:
        score += 0.10
        matches.append("short_duration")
    if after_date and word_count <= config.title_max_words:
        score += 0.08
        matches.append("post_date_position")

    if not context_match:
        score = min(score, 0.56 if after_date else 0.42)
    return min(score, 0.95), matches


def _words_until(words: list[TranscriptWord], end_sec: float) -> list[TranscriptWord]:
    return [word for word in words if word.end_sec <= end_sec + 0.001]


def _words_after(words: list[TranscriptWord], start_sec: float) -> list[TranscriptWord]:
    return [word for word in words if word.start_sec >= start_sec - 0.001]


def _join_words(words: Iterable[TranscriptWord]) -> str:
    return " ".join(word.word.strip() for word in words if word.word.strip()).strip()


def _segment_after_time(segment: TranscriptSegment, start_sec: float) -> TranscriptSegment | None:
    if segment.end_sec <= start_sec + 0.001:
        return None

    remaining_words = [word for word in segment.words if word.end_sec > start_sec + 0.001]
    if segment.words and not remaining_words:
        return None
    if remaining_words:
        return TranscriptSegment(
            text=_join_words(remaining_words),
            start_sec=remaining_words[0].start_sec,
            end_sec=remaining_words[-1].end_sec,
            avg_logprob=segment.avg_logprob,
            no_speech_prob=segment.no_speech_prob,
            words=remaining_words,
        )

    return TranscriptSegment(
        text=segment.text,
        start_sec=max(segment.start_sec, start_sec),
        end_sec=segment.end_sec,
        avg_logprob=segment.avg_logprob,
        no_speech_prob=segment.no_speech_prob,
        words=[],
    )


def _classify_remaining_segments(
    segments: list[TranscriptSegment],
    *,
    date_detected: bool,
    context_phrases: list[str],
    config: IntroClassifierConfig,
) -> list[IntroSegmentClassification]:
    classified: list[IntroSegmentClassification] = []
    content_started = False

    for segment in segments:
        if not segment.text.strip():
            continue

        title_score, title_matches = _score_title_segment(
            segment,
            context_phrases=context_phrases,
            config=config,
            repeated_phrases=config.repeated_phrases,
            boilerplate_phrases=config.boilerplate_phrases,
            after_date=date_detected and not content_started,
        )
        if not content_started and title_score >= config.title_confidence_threshold:
            classified.append(
                IntroSegmentClassification(
                    label="title",
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    text=segment.text,
                    confidence=round(title_score, 3),
                    removable=False,
                    matched_patterns=sorted(set(title_matches)),
                    reason="Opening speech matched a title-like phrase and is preserved.",
                )
            )
            continue

        content_started = True
        classified.append(
            IntroSegmentClassification(
                label="content",
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text=segment.text,
                confidence=0.85,
                removable=False,
                reason="Opening content speech starts here and is preserved.",
            )
        )

    return classified


def detect_intro(
    transcript: TranscriptResult,
    *,
    config: IntroClassifierConfig,
    context_phrases: list[str] | None = None,
) -> IntroDetectionResult:
    context_phrases = context_phrases or []
    snippet = transcript.text[:200]
    if not transcript.text.strip():
        return IntroDetectionResult(
            detected=False,
            confidence=0.0,
            reason="No opening transcript was available.",
            transcript_snippet=snippet,
        )

    words = _collect_words(transcript)
    opening_segments = [
        segment
        for segment in transcript.segments
        if segment.start_sec <= config.max_auto_trim_start_sec
    ]
    matched_patterns: list[str] = []
    classified_segments: list[IntroSegmentClassification] = []

    opening_date_end = _match_end_time_from_words(_OPENING_DATE_REGEXES, words, max_start_chars=4)
    opening_date_words = _words_until(words, opening_date_end) if opening_date_end is not None else []
    opening_date_text = _join_words(opening_date_words)
    opening_date_score, opening_date_matches = _score_date_text(
        opening_date_text,
        config.repeated_phrases,
        config.boilerplate_phrases,
    )
    if opening_date_text:
        matched_patterns.extend(opening_date_matches)

    segment_date_candidates: list[TranscriptSegment] = []
    if not opening_date_text:
        for segment in opening_segments:
            segment_score, _segment_matches = _score_date_text(
                segment.text,
                config.repeated_phrases,
                config.boilerplate_phrases,
            )
            if not segment_date_candidates and segment_score < config.date_segment_min_score:
                break
            if segment_score >= config.date_segment_min_score:
                segment_date_candidates.append(segment)
                continue
            break

        if segment_date_candidates:
            segment_text = " ".join(segment.text for segment in segment_date_candidates)
            opening_date_score, opening_date_matches = _score_date_text(
                segment_text,
                config.repeated_phrases,
                config.boilerplate_phrases,
            )
            matched_patterns.extend(opening_date_matches)
            if opening_date_score >= config.confidence_threshold:
                opening_date_end = segment_date_candidates[-1].end_sec
                opening_date_text = segment_text

    detected = (
        opening_date_end is not None
        and opening_date_end <= config.max_auto_trim_start_sec
        and opening_date_score >= config.confidence_threshold
    )

    if detected:
        date_start_sec = opening_date_words[0].start_sec if opening_date_words else opening_segments[0].start_sec
        classified_segments.append(
            IntroSegmentClassification(
                label="date",
                start_sec=date_start_sec,
                end_sec=opening_date_end,
                text=opening_date_text or transcript.text,
                confidence=round(opening_date_score, 3),
                removable=True,
                matched_patterns=sorted(set(opening_date_matches)),
                reason="Opening speech matched a spoken calendar date and is removable.",
            )
        )

    remaining_segments: list[TranscriptSegment] = []
    split_from_sec = opening_date_end or 0.0
    for segment in opening_segments:
        remaining_segment = _segment_after_time(segment, split_from_sec) if detected else segment
        if remaining_segment is not None:
            remaining_segments.append(remaining_segment)
    if detected:
        remaining_segments = [
            segment
            for segment in remaining_segments
            if segment.start_sec >= (opening_date_end or 0.0) - 0.001
        ]

    classified_segments.extend(
        _classify_remaining_segments(
            remaining_segments,
            date_detected=detected,
            context_phrases=context_phrases,
            config=config,
        )
    )

    matched_patterns.extend(
        pattern
        for segment in classified_segments
        for pattern in segment.matched_patterns
    )
    matched_patterns = sorted(set(matched_patterns))

    title_segments = [segment for segment in classified_segments if segment.label == "title"]
    content_segments = [segment for segment in classified_segments if segment.label == "content"]
    title_start_sec = title_segments[0].start_sec if title_segments else None
    title_end_sec = title_segments[-1].end_sec if title_segments else None
    content_start_sec = content_segments[0].start_sec if content_segments else None
    confidence = round(
        min(
            0.99,
            max(
                opening_date_score if detected else 0.0,
                max((segment.confidence for segment in title_segments), default=0.0),
            ),
        ),
        3,
    )

    if not detected:
        return IntroDetectionResult(
            detected=False,
            trim_offset_sec=opening_date_end,
            confidence=confidence,
            reason="Opening transcript did not start with a removable spoken date.",
            matched_patterns=matched_patterns,
            transcript_snippet=snippet,
            classified_segments=classified_segments,
            evidence={
                "date_end_sec": opening_date_end,
                "title_start_sec": title_start_sec,
                "title_end_sec": title_end_sec,
                "content_start_sec": content_start_sec,
            },
        )

    return IntroDetectionResult(
        detected=True,
        trim_offset_sec=opening_date_end,
        confidence=confidence,
        reason="Opening transcript was classified as date, title, and content; only the date span is removable.",
        matched_patterns=matched_patterns,
        transcript_snippet=snippet,
        classified_segments=classified_segments,
        evidence={
            "date_end_sec": opening_date_end,
            "title_start_sec": title_start_sec,
            "title_end_sec": title_end_sec,
            "content_start_sec": content_start_sec,
        },
    )
