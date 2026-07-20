# megatensors

`megatensors` is the public Python SDK for MEGA, modeled around the familiar
Hugging Face Hub developer experience while targeting MEGA repositories,
datasets, Spaces, jobs, inference, storage, and Xet-backed large files. Move
from a familiar Hub workflow to MEGA without changing how your team thinks
about discovery, versioned artifacts, or programmatic automation.

`megatensors` is an independent SDK. It is not affiliated with or endorsed by
Hugging Face; compatibility claims apply only where the documented API surface
explicitly says so.

## Install

```bash
pip install megatensors
```

For local development, install from this checkout with `pip install -e .`.

## Scope

- `megatensors/` contains the public SDK, Hub-compatible client surface, CLI,
  tensor loading utilities, and framework adapters.
- `tests/` contains the client and storage contract coverage.
- Infrastructure credentials and deployment configuration are intentionally
  kept out of this public repository.

The package is MIT licensed. Service implementations live in private MEGA
organization repositories; public users interact through this SDK and the
documented MEGA API.
