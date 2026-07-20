<p align="center">
  <a href="https://mega.tensorplay.cn/">
    <img src="https://mega.tensorplay.cn/assets/logo-D1t6EjrA.webp" alt="MEGA" width="420" />
  </a>
</p>

<p align="center"><i>The public Python SDK for building, publishing, and running with MEGA.</i></p>

<p align="center">
  <a href="https://mega.tensorplay.cn/docs/hub/sdk"><img alt="Documentation" src="https://img.shields.io/website?url=https%3A%2F%2Fmega.tensorplay.cn%2Fdocs%2Fhub%2Fsdk&label=docs"></a>
  <a href="https://github.com/ohtensorplay/megatensors/actions/workflows/package.yml"><img alt="Package" src="https://github.com/ohtensorplay/megatensors/actions/workflows/package.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/ohtensorplay/megatensors/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/ohtensorplay/megatensors"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="https://github.com/ohtensorplay/megatensors/blob/main/LICENSE"><img alt="Apache-2.0 License" src="https://img.shields.io/github/license/ohtensorplay/megatensors"></a>
</p>

---

**Documentation:** <https://mega.tensorplay.cn/docs/hub/sdk><br/>
**Source:** <https://github.com/ohtensorplay/megatensors>

---

## Welcome to megatensors

`megatensors` connects Python applications to MEGA repositories, datasets,
Spaces, Jobs, inference, storage, and Xet-backed large files. It combines a
high-level Hub client with efficient tensor loading and framework adapters, so
the same package can move an artifact, inspect it, and load it for execution.

## Key features

- Discover and manage models, datasets, Spaces, and versioned repositories.
- Download individual files or complete snapshots with a local cache.
- Upload files and folders with resumable large-file support.
- Run inference, remote Jobs, schedules, and Sandbox sessions.
- Load tensors into PyTorch and Paddle through a shared API.
- Sign and verify artifacts for reproducible distribution workflows.

## Installation

Install the current public source release:

```bash
pip install "megatensors @ git+https://github.com/ohtensorplay/megatensors.git@main"
```

For local development:

```bash
git clone https://github.com/ohtensorplay/megatensors.git
cd megatensors
pip install -e .
```

## Quick start

Download a repository snapshot:

```python
from megatensors import snapshot_download

local_path = snapshot_download("owner/model-name")
print(local_path)
```

Use the typed Hub client:

```python
from megatensors import MegaApi

api = MegaApi()
models = api.list_models(search="embedding")
for model in models:
    print(model.id)
```

Explore the CLI:

```bash
mega --help
```

## Repository layout

- `megatensors/` — SDK, CLI, Hub client, tensor loaders, and framework adapters.
- `tests/` — API, storage, CLI, and compatibility contracts.
- `examples/` — runnable loading, generation, and signing examples.

## Contributing

Issues and pull requests are welcome. Run the focused tests for the area you
change and keep credentials, deployment configuration, and generated artifacts
out of the repository.

## License

Apache-2.0
