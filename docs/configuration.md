# Configuration Guide

Most applications should use the public API directly:

```python
from megatensors import mega_open

with mega_open("model.mega", device="cuda:0") as artifact:
    tensor = artifact.get_tensor("layers.0.weight")
```

Backend configuration remains available for integrations that need to select
storage backends such as GDS, no-GDS, unified memory, DirectStorage, or 3FS.
Configuration is discovered in this priority order:

1. `MEGATENSORS_CONFIG=/path/to/config.json`
2. `./megatensors.json` in the working directory
3. Built-in defaults

Default configuration:

```json
{
  "loader": "base",
  "framework": "pytorch",
  "parallel": {
    "use_pipeline": false
  },
  "debug": {
    "debug_log": false,
    "set_numa": true,
    "disable_cache": true
  }
}
```

3FS backend configuration:

```json
{
  "loader": "3fs",
  "3fs": {
    "mount_point": "/mnt/3fs",
    "entries": 64,
    "io_depth": 0,
    "buffer_size": 67108864
  }
}
```

Pipeline queue semantics:

| `queue_size` | Mode | GPU Memory | Behavior |
|---|---|---|---|
| `-1` | serial | 1 batch | copy then broadcast |
| `0` | unbuffered pipeline | up to 2 batches | copy overlaps broadcast |
| `>0` | buffered pipeline | up to `queue_size+1` batches | producer fills queue |
