Overview
=========

# MEGA Artifact Loading

MEGA is a model artifact format and loader stack for high-performance inference
startup. The public Python API is intentionally small:

- `mega_open(...)` for lazy tensor access.
- `iter_tensors(...)`, `load_tensor(...)`, and `load_state_dict(...)` for
  PyTorch-style state loading.
- `load_model(...)` for constructing a model object and loading weights.
- `load_tokenizer(...)` for constructing a Hugging Face tokenizer object from
  typed MEGA KV metadata.
- `open_kv_cache(...)`, `load_kv_tensor(...)`, and `write_kv_cache(...)` for
  `.megakv` sidecars.
- `append_footer_overlay(...)` for metadata-only append updates without
  rewriting tensor payload.

Except for `mega_open`, public function names avoid a redundant format prefix.

# Basic Usage

```python
from megatensors import mega_open

with mega_open("model.mega", nogds=True, device="cpu") as artifact:
    for name in artifact.keys():
        tensor = artifact.get_tensor(name).clone()
```

```python
from megatensors import load_model, load_state_dict, load_tokenizer

state = load_state_dict("model.mega", device="cpu", nogds=True)
tokenizer = load_tokenizer("model.mega")
model = load_model("model.mega", device="cuda:0")
```

`load_model` uses `model.class` and `model.init.*` metadata when present. If an
artifact does not carry model construction metadata, pass `model_class` and
`model_kwargs`.

# Hub CLI

## Hub Python API

The MEGA Hub client is the migrated and customized Hugging Face Hub cache stack,
backed by Cloudflare Workers, D1, and R2 at `https://mega.tensorplay.cn`. It
keeps content-addressed snapshots, file locks, retry logic, interrupted-download
resume, filtered snapshots, and an fsspec filesystem.

```python
from megatensors import MegaFileSystem, mega_hub_download, snapshot_download

artifact = mega_hub_download("mega/example", "model.mega")
snapshot = snapshot_download("mega/example", allow_patterns=["*.mega", "*.json"])
with MegaFileSystem().open("mega://mega/example/model.mega", "rb") as file:
    header = file.read(4096)
```

Use `MEGA_ENDPOINT`, `MEGA_TOKEN`, `MEGA_HOME`, `MEGA_HUB_CACHE`, and
`MEGA_HUB_OFFLINE` to configure the endpoint, credentials, cache, and offline mode.
Uploads use the HF-compatible commit protocol. Files larger than 20 MiB are
chunked and deduplicated by the official `hf_xet` runtime, stored on MEGA's
registered native Xet CAS, and represented in Git by canonical LFS pointers.
The `mega` and `hf` CLIs share hashes and can read each other's uploads.

`mega login` opens the Cloudflare Access-protected device authorization page
and waits for approval. `mega login --token "$MEGA_TOKEN"` remains the
non-interactive path for CI.

The `mega` CLI keeps the local artifact commands and adds Hub-style workflows
similar to Hugging Face Hub and ModelScope:

```bash
mega auth login --endpoint https://mega.tensorplay.cn --token "$MEGA_TOKEN"
mega repos create mega/qwen --exist-ok
mega upload mega/qwen ./Qwen3.5-0.8B/mega --include "*.mega" --include "*.json"
mega repos files mega/qwen
mega repos history mega/qwen
mega discussions create mega/qwen --title "Serving notes" --body-file notes.md
mega discussions list mega/qwen --format json
mega snapshot mega/qwen --local-dir ./downloaded
```

The default backend contract is implemented in `mega-hub` with
Cloudflare Workers, R2, and D1.

# KV Cache Sidecars

KV cache is stored outside the model artifact:

```python
from megatensors import load_kv_tensor, open_kv_cache, write_kv_cache

write_kv_cache(
    "prompt.megakv",
    [
        {
            "name": "kv.layer.0.k",
            "layer_id": 0,
            "kv_role": 1,
            "sequence_id": 0,
            "token_start": 0,
            "token_count": 2,
            "shape": [1, 2, 2],
            "logical_dtype": "F32",
            "data": payload_bytes,
        }
    ],
    metadata={
        "mega.kv.model_payload_hash": payload_hash,
        "mega.kv.tokenizer_hash": tokenizer_hash,
    },
)

cache = open_kv_cache("prompt.megakv")
tensor = load_kv_tensor("prompt.megakv", "kv.layer.0.k")
```

The writer and parser are implemented in the C++ extension. Python only
normalizes user objects and forwards to the backend.

# Footer Overlay

Metadata-only updates can be appended without touching tensor payload bytes:

```python
from megatensors import append_footer_overlay

append_footer_overlay(
    "model.mega",
    {
        "mega.hash.payload.sha256": payload_sha256,
        "mega.source.model_id": "publisher/model",
    },
    generation=2,
)
```

The C++ parser validates the overlay checksum and rejects structural overrides
such as tensor directory counts or alignment.

# Integrity And Trust

MEGA treats input files as untrusted. The parser validates offsets, sizes,
dtype/shape consistency, codec fields, duplicate names, and checksums. Payload
CRC32/SHA-256 verification is explicit so lazy opens stay fast:

```python
from megatensors.common import MegaTensorsMetadata
from megatensors.frameworks import get_framework_op

meta = MegaTensorsMetadata.from_file("model.mega", get_framework_op("pt"))
meta.verify_payload_sha256()
trust = meta.verify_trust(trusted_roots_pem, allowed_publishers={"publisher"})
```

Trust roots are supplied by caller policy. Certificates embedded in an artifact
are never trusted as roots.

# Backends

The loader stack includes GDS, no-GDS, unified-memory, DirectStorage, and 3FS
backends. These are backend concerns; most users should start from the public
API above and select behavior with arguments such as `nogds=True` or with the
configuration file documented in `configuration.md`.
