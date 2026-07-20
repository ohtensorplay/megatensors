# megatensors

`megatensors` is the public Python SDK for MEGA. Build reproducible ML
workflows around versioned repositories, datasets, Spaces, jobs, inference,
storage, and Xet-backed large files—all from a single, typed client.

## Build with MEGA

- Discover and publish models, datasets, and applications programmatically.
- Move large artifacts reliably with resumable transfers and storage-aware
  tooling.
- Automate inference, remote Jobs, and Spaces without coupling code to
  deployment credentials.

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
