import os
import sys
from types import SimpleNamespace

from megatensors._hub.utils._xet import XetSessionHolder


def test_xet_session_uses_four_initial_uploads_by_default(monkeypatch):
    observed = []

    class FakeXetSession:
        def __init__(self):
            observed.append((
                os.environ.get("HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY"),
                os.environ.get("HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY"),
                os.environ.get("HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"),
            ))

    monkeypatch.delenv("HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY", raising=False)
    monkeypatch.delenv("HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY", raising=False)
    monkeypatch.delenv("HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY", raising=False)
    monkeypatch.setitem(sys.modules, "hf_xet", SimpleNamespace(XetSession=FakeXetSession))

    session = XetSessionHolder().get()

    assert isinstance(session, FakeXetSession)
    assert observed == [("4", "8", "8")]


def test_xet_session_preserves_an_explicit_upload_concurrency(monkeypatch):
    observed = []

    class FakeXetSession:
        def __init__(self):
            observed.append((
                os.environ.get("HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY"),
                os.environ.get("HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY"),
                os.environ.get("HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"),
            ))

    monkeypatch.setenv("HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY", "6")
    monkeypatch.setenv("HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY", "7")
    monkeypatch.setenv("HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY", "5")
    monkeypatch.setitem(sys.modules, "hf_xet", SimpleNamespace(XetSession=FakeXetSession))

    XetSessionHolder().get()

    assert observed == [("6", "7", "5")]
