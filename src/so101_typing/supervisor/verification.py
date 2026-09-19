from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import math


class ScreenVerificationStatus(StrEnum):
    CONFIRMED_SUCCESS = "CONFIRMED_SUCCESS"
    CONFIRMED_NO_CHANGE = "CONFIRMED_NO_CHANGE"
    CONFIRMED_WRONG = "CONFIRMED_WRONG"
    UNCERTAIN = "UNCERTAIN"


class FrameEvidenceKind(StrEnum):
    SUCCESS = "success"
    NO_CHANGE = "no_change"
    WRONG = "wrong"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class FrameEvidence:
    kind: FrameEvidenceKind
    observed_text: str
    matched_line: str | None
    matched_window: str | None
    prefix_distance: int | None
    continuation: str | None


@dataclass(frozen=True, slots=True)
class ScreenVerificationResult:
    status: ScreenVerificationStatus
    total_frames: int
    usable_frames: int
    success_votes: int
    no_change_votes: int
    wrong_votes: int
    uncertain_votes: int
    wrong_character: str | None
    winning_votes: int
    vote_fraction: float
    frame_evidence: tuple[FrameEvidence, ...]


def _compact_alnum(text: str) -> str:
    return "".join(
        character
        for character in str(text).upper()
        if character.isascii() and character.isalnum()
    )


def _compact_lines(text: str) -> list[str]:
    lines: list[str] = []

    for raw_line in str(text).splitlines():
        compact = _compact_alnum(raw_line)

        if compact:
            lines.append(compact)

    return lines


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0

    if not left:
        return len(right)

    if not right:
        return len(left)

    previous = list(range(len(right) + 1))

    for i, left_char in enumerate(left, start=1):
        current = [i]

        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = (
                previous[j - 1]
                + (0 if left_char == right_char else 1)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def classify_frame_text(
    observed_text: str,
    *,
    confirmed_prefix: str,
    expected_char: str,
    max_prefix_distance: int = 2,
) -> FrameEvidence:
    prefix = _compact_alnum(confirmed_prefix)
    expected = _compact_alnum(expected_char)

    if not prefix:
        raise ValueError(
            "confirmed_prefix must contain at least one alphanumeric character"
        )

    if len(expected) != 1:
        raise ValueError(
            "expected_char must contain exactly one ASCII alphanumeric character"
        )

    if max_prefix_distance < 0:
        raise ValueError(
            "max_prefix_distance must be non-negative"
        )

    best = None

    prefix_length = len(prefix)

    for line in _compact_lines(observed_text):
        for start in range(len(line)):
            for length_delta in (0, -1, 1):
                window_length = (
                    prefix_length
                    + length_delta
                )

                if window_length <= 0:
                    continue

                end = start + window_length

                if end > len(line):
                    continue

                window = line[start:end]

                distance = _levenshtein(
                    prefix,
                    window,
                )

                candidate = (
                    distance,
                    abs(length_delta),
                    start,
                    line,
                    window,
                    end,
                )

                if best is None or candidate[:3] < best[:3]:
                    best = candidate

    if best is None:
        return FrameEvidence(
            kind=FrameEvidenceKind.UNCERTAIN,
            observed_text=observed_text,
            matched_line=None,
            matched_window=None,
            prefix_distance=None,
            continuation=None,
        )

    (
        distance,
        _,
        _,
        line,
        window,
        end,
    ) = best

    if distance > max_prefix_distance:
        return FrameEvidence(
            kind=FrameEvidenceKind.UNCERTAIN,
            observed_text=observed_text,
            matched_line=line,
            matched_window=window,
            prefix_distance=distance,
            continuation=None,
        )

    continuation = (
        line[end]
        if end < len(line)
        else None
    )

    if continuation is None:
        kind = FrameEvidenceKind.NO_CHANGE

    elif continuation == expected:
        kind = FrameEvidenceKind.SUCCESS

    else:
        kind = FrameEvidenceKind.WRONG

    return FrameEvidence(
        kind=kind,
        observed_text=observed_text,
        matched_line=line,
        matched_window=window,
        prefix_distance=distance,
        continuation=continuation,
    )


def verify_screen_texts(
    observed_texts: Sequence[str],
    *,
    confirmed_prefix: str,
    expected_char: str,
    min_vote_fraction: float = 0.60,
    max_prefix_distance: int = 2,
) -> ScreenVerificationResult:
    if not observed_texts:
        raise ValueError(
            "at least one observed text is required"
        )

    if (
        not math.isfinite(min_vote_fraction)
        or not 0.0 < min_vote_fraction <= 1.0
    ):
        raise ValueError(
            "min_vote_fraction must be in (0, 1]"
        )

    evidence = tuple(
        classify_frame_text(
            text,
            confirmed_prefix=confirmed_prefix,
            expected_char=expected_char,
            max_prefix_distance=max_prefix_distance,
        )
        for text in observed_texts
    )

    success_votes = sum(
        item.kind is FrameEvidenceKind.SUCCESS
        for item in evidence
    )

    no_change_votes = sum(
        item.kind is FrameEvidenceKind.NO_CHANGE
        for item in evidence
    )

    wrong_items = [
        item
        for item in evidence
        if (
            item.kind is FrameEvidenceKind.WRONG
            and item.continuation is not None
        )
    ]

    wrong_counter = Counter(
        item.continuation
        for item in wrong_items
    )

    if wrong_counter:
        wrong_character, wrong_votes = (
            wrong_counter.most_common(1)[0]
        )
    else:
        wrong_character = None
        wrong_votes = 0

    uncertain_votes = sum(
        item.kind is FrameEvidenceKind.UNCERTAIN
        for item in evidence
    )

    total = len(evidence)

    usable_frames = (
        total - uncertain_votes
    )

    required_votes = math.ceil(
        total * min_vote_fraction
    )

    if success_votes >= required_votes:
        status = (
            ScreenVerificationStatus.CONFIRMED_SUCCESS
        )
        winning_votes = success_votes

    elif no_change_votes >= required_votes:
        status = (
            ScreenVerificationStatus.CONFIRMED_NO_CHANGE
        )
        winning_votes = no_change_votes

    elif wrong_votes >= required_votes:
        status = (
            ScreenVerificationStatus.CONFIRMED_WRONG
        )
        winning_votes = wrong_votes

    else:
        status = ScreenVerificationStatus.UNCERTAIN

        winning_votes = max(
            success_votes,
            no_change_votes,
            wrong_votes,
        )

    return ScreenVerificationResult(
        status=status,
        total_frames=total,
        usable_frames=usable_frames,
        success_votes=success_votes,
        no_change_votes=no_change_votes,
        wrong_votes=wrong_votes,
        uncertain_votes=uncertain_votes,
        wrong_character=wrong_character,
        winning_votes=winning_votes,
        vote_fraction=(
            winning_votes / total
        ),
        frame_evidence=evidence,
    )
