# SPDX-License-Identifier: Apache-2.0

import io
import time

from megatensors._hub.utils import _terminal


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_spinner_animates_and_clears_the_tty_line(monkeypatch):
    stream = _TTYBuffer()
    monkeypatch.setattr(_terminal.sys, "stderr", stream)

    with _terminal.Spinner("Preparing browser authorization...", interval=0.002):
        time.sleep(0.01)

    output = stream.getvalue()
    assert "Preparing browser authorization..." in output
    assert any(frame in output for frame in _terminal.Spinner._frames)
    assert output.endswith("\r\033[K")


def test_spinner_prints_one_status_line_when_stderr_is_redirected(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(_terminal.sys, "stderr", stream)

    with _terminal.Spinner("Preparing browser authorization..."):
        pass

    assert stream.getvalue() == "Preparing browser authorization...\n"
