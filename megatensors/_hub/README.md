# MEGA Hub client

`megatensors._hub` is the high-level Python client used by the `mega` CLI. It
targets the public MEGA API and shares the same authentication, repository,
revision, transfer, and cache semantics.

## Download and cache

Use `mega_hub_download()` for one file and `snapshot_download()` for a complete
revision. Both use the content-addressed cache under `MEGA_HUB_CACHE`; branch and
tag names resolve to immutable revisions before files are materialized.

```python
from megatensors import mega_hub_download, snapshot_download

config = mega_hub_download("mega/demo", "config.json", revision="main")
snapshot = snapshot_download("mega/demo", revision="main")
```

MEGA repository URIs use the `mega://` scheme:

```text
mega://models/namespace/repository@revision/path
mega://datasets/namespace/repository@revision/path
mega://spaces/namespace/repository@revision/path
```

## Repository API

`MegaApi` provides the high-level repository workflow. Python and CLI
operations share the same authenticated client and protocol behavior.

```python
from megatensors import MegaApi

api = MegaApi()
api.create_repo("namespace/repository", repo_type="model", private=True)
api.upload_file(
    repo_id="namespace/repository",
    path_or_fileobj="model.safetensors",
    path_in_repo="model.safetensors",
    commit_message="Upload model",
)
info = api.repo_info("namespace/repository", files_metadata=True)
```

Atomic commits stage objects first and publish add, copy, and delete operations
in one compare-and-swap update. Folder uploads and `--sync` use the same commit
primitive.

## Repository community

The community CLI keeps the familiar Hugging Face discussions vocabulary.
Markdown can be provided inline, from a file, or from standard input; JSON and
quiet output use the same shared CLI output layer as repository commands.

```bash
mega discussions list namespace/repository
mega discussions create namespace/repository --title "Runtime question" --body-file question.md
mega discussions comment namespace/repository 3 --body-file -
mega discussions react namespace/repository 3 <message-id>
mega discussions diff namespace/repository 4 --format json
mega discussions close namespace/repository 3 --comment "Resolved." --yes
```

Pull requests are branch-backed and therefore require a source branch:

```bash
mega discussions create namespace/repository \
  --title "Ship tokenizer update" \
  --pull-request \
  --source-branch feature/tokenizer \
  --target-branch main
```

Fine-grained tokens need `repo:read` for private repository visibility and
`community:write` for create, reply, edit, reaction, status, merge, and delete
operations.

## Authentication

Use `MEGA_TOKEN` or the active token saved by the CLI:

```bash
mega auth login
mega auth whoami
mega auth logout
```

The endpoint defaults to `https://mega.tensorplay.cn` and can be overridden with
`MEGA_ENDPOINT`.
