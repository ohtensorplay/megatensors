from __future__ import annotations

from types import SimpleNamespace

from megatensors._hub import _upload_pipeline as upload_pipeline


class _RegularOperation:
    def __init__(self, path: str, size: int) -> None:
        self.path_in_repo = path
        self.upload_info = SimpleNamespace(size=size)
        self._should_ignore = False
        self._upload_mode = "regular"


class _FakeApi:
    endpoint = "https://hub.example"

    def _build_mega_headers(self, *, token):
        return {"authorization": f"Bearer {token}"}


def test_regular_upload_batches_stay_within_hub_commit_body_budget(monkeypatch):
    mib = 1024 * 1024
    operations = [
        _RegularOperation("first.bin", 12 * mib),
        _RegularOperation("second.bin", 12 * mib),
    ]
    batches = []

    monkeypatch.setattr(upload_pipeline, "get_xet_session", object)
    monkeypatch.setattr(upload_pipeline, "_fetch_upload_modes", lambda **kwargs: None)
    pipeline = upload_pipeline._UploadPipeline(
        _FakeApi(),
        repo_id="mega/demo",
        repo_type="model",
        add_operations=operations,
        delete_operations=[],
        commit_message="Upload files",
        commit_description=None,
        token="token",
        revision="main",
        create_pr=False,
        parent_commit=None,
    )
    monkeypatch.setattr(
        pipeline,
        "_enqueue",
        lambda batch: batches.append(batch) if batch.ops else None,
    )

    pipeline._coordinator_loop()

    assert [[op.path_in_repo for op in batch.ops] for batch in batches] == [
        ["first.bin"],
        ["second.bin"],
    ]
    assert all(
        batch.regular_bytes <= upload_pipeline.REGULAR_CONTENT_BYTES_BUDGET
        for batch in batches
    )
    max_base64_bytes = 4 * ((upload_pipeline.REGULAR_CONTENT_BYTES_BUDGET + 2) // 3)
    assert max_base64_bytes < 32 * mib
