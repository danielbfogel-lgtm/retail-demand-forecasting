"""Pinned requirements (US-34, PRD §40, §55).

Reproducibility needs every dependency nailed to one exact version — a range like ``pandas>=2.2``
could quietly resolve to a different release on a different machine or a different day and change
a result. This file checks ``requirements.txt`` uses ``==`` everywhere, and that the installed
environment has no conflicting packages (``pip check``).

The one line that is not a ``==`` pin is ``-e .``, the editable install of this project's own
src-layout packages. It carries no version to pin — it is this working tree — and it is what makes
Streamlit Community Cloud, which installs ``requirements.txt`` and nothing else, able to import
``app`` and ``pipeline``. It is asserted for separately below.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest

from pipeline import paths

REQUIREMENTS_FILE = paths.PROJECT_ROOT / "requirements.txt"
_PIN_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-]+$")
_SELF_INSTALL = "-e ."  # this project, editable; no version exists to pin


def _requirement_lines() -> list[str]:
    text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def test_requirements_file_is_not_empty() -> None:
    assert _requirement_lines(), f"{REQUIREMENTS_FILE} has no dependency lines"


def test_every_requirement_uses_double_equals_pin() -> None:
    unpinned = [
        line
        for line in _requirement_lines()
        if line != _SELF_INSTALL and not _PIN_PATTERN.match(line)
    ]
    assert unpinned == [], f"requirements.txt has non-== pin(s): {unpinned}"


def test_requirements_installs_this_project_editable() -> None:
    """``-e .`` is present and last.

    Streamlit Community Cloud installs ``requirements.txt`` and runs nothing else, so without this
    line the deployed app cannot import ``app`` or ``pipeline``. Last, so it resolves after the
    pinned dependencies rather than dragging its own resolution in first.
    """
    lines = _requirement_lines()
    assert _SELF_INSTALL in lines, f"requirements.txt is missing the {_SELF_INSTALL!r} line"
    assert lines[-1] == _SELF_INSTALL, f"{_SELF_INSTALL!r} must be the last requirement line"


def test_pip_check_passes() -> None:
    """The installed environment has no version conflicts.

    CI (US-35) installs with plain ``pip``, so ``python -m pip check`` is the real check there.
    Locally (CLAUDE.md §4) the venv is built with ``uv``, which does not install a ``pip`` module
    into it — ``uv pip check`` is the equivalent for that environment. Either way this asserts an
    actual conflict check ran; it never silently skips just because ``pip`` itself is absent.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 and "No module named pip" in (result.stdout + result.stderr):
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("neither pip nor uv is available to run a dependency-conflict check")
        result = subprocess.run(
            [uv, "pip", "check", "--python", sys.executable],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
