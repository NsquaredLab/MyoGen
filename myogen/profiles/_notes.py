"""Notes on the back of a profile card.

They are ``ProfileNote`` warnings, printed at most once per key per Python process, so a run's console shows each
note once. ``warnings.simplefilter("ignore", ProfileNote)`` silences them; ``"error"`` turns them into errors.
"""
from __future__ import annotations

import sys
import warnings

_shown: set[str] = set()


class ProfileNote(UserWarning):
    """A note from a profile card."""


def _caller_stacklevel() -> int:
    """The ``stacklevel`` (for a ``warnings.warn`` in ``emit_note``) of the first frame outside MyoGen."""
    level, frame = 2, sys._getframe(2)
    while frame is not None and (str(frame.f_globals.get("__name__")).split(".")[0] == "myogen"
                                 or frame.f_code.co_filename.startswith("<@beartype")):  # type-check wrappers
        level, frame = level + 1, frame.f_back
    return level


def emit_note(key: str, text: str) -> None:
    """Warn ``text`` as a ``ProfileNote`` the first time ``key`` is seen in this process, at the user's line."""
    if key in _shown:
        return
    _shown.add(key)
    warnings.warn(text, ProfileNote, stacklevel=_caller_stacklevel())


def reset_notes() -> None:
    """Forget which notes were shown (tests)."""
    _shown.clear()
