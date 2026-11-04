"""Notes on the back of a profile card, and the hint for components built without a card.

Both are ``ProfileNote`` warnings, printed at most once per key per Python process, so a run's console shows each
note once. ``warnings.simplefilter("ignore", ProfileNote)`` silences them; ``"error"`` turns them into errors.
"""
from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

_shown: set[str] = set()
_in_card_build: ContextVar[bool] = ContextVar("myogen_profile_card_build", default=False)


class ProfileNote(UserWarning):
    """A note from a profile card, or the hint that a component was built without one."""


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


def hand_built(component: str) -> None:
    """Called at the end of the constructors a card builds; hints once per process when no card is building."""
    if _in_card_build.get():
        return
    emit_note("hand_built", f"no profile used for {component}; see myogen.load_profile('fdi_semg') for the "
                            "validated FDI surface-EMG settings (this hint is shown once)")


@contextmanager
def building_from_card():
    """Constructors called inside this block belong to a card build: no hand-built hint."""
    token = _in_card_build.set(True)
    try:
        yield
    finally:
        _in_card_build.reset(token)


def reset_notes() -> None:
    """Forget which notes were shown (tests)."""
    _shown.clear()
