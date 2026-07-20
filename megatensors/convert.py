# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import cpp as megacpp


MEGA_COMPRESSION_NONE = 0
MEGA_COMPRESSION_ZSTD = 1
MEGA_COMPRESSION_AUTO = 0xFFFFFFFF
MEGA_COMPRESSION_BY_NAME = {
    "none": MEGA_COMPRESSION_NONE,
    "zstd": MEGA_COMPRESSION_ZSTD,
    "auto": MEGA_COMPRESSION_AUTO,
}
SAFETENSORS_DTYPE_TO_MEGA = {
    "BOOL": "BOOL",
    "U8": "U8",
    "I8": "I8",
    "I16": "I16",
    "I32": "I32",
    "I64": "I64",
    "F16": "F16",
    "BF16": "BF16",
    "F32": "F32",
    "F64": "F64",
    "F8_E5M2": "F8_E5M2",
    "F8_E4M3": "F8_E4M3",
}


@dataclass(frozen=True)
class ConvertResult:
    index_path: Path
    shard_paths: list[Path]
    tensor_count: int
    payload_nbytes: int
    elapsed_seconds: float


def convert_model(
    model_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    basename: str = "model",
    max_shard_size: int | str = "5GB",
    alignment: int = 4096,
    metadata: Mapping[str, Any] | None = None,
    shard_metadata: str = "first",
    compression: str = "none",
    compression_level: int = 3,
    compression_min_ratio: float = 0.98,
    signing: Any = None,
) -> ConvertResult:
    """Convert a safetensors model directory to sharded MEGA artifacts.

    Tensor payload copying is delegated to the C++ backend. Python only parses
    safetensors headers and model/tokenizer metadata, then assigns tensors to
    shards on tensor boundaries.
    """
    src_dir = Path(model_dir).resolve()
    dst_dir = Path(output_dir).resolve() if output_dir is not None else src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = _parse_size(max_shard_size)
    compression_codec = _compression_codec(compression)
    started = time.perf_counter()
    tensors, total_payload = build_tensor_table(src_dir)
    shards = _plan_shards(tensors, max_bytes)
    global_metadata = {
        **_model_metadata(src_dir, len(tensors), total_payload, alignment),
        **dict(metadata or {}),
    }
    global_metadata.update(_tokenizer_metadata(src_dir))
    minimal_metadata = _minimal_shard_metadata(global_metadata, alignment)

    shard_paths: list[Path] = []
    total_shards = len(shards)
    for shard_index, shard_tensors in enumerate(shards, start=1):
        shard_name = f"{basename}-{shard_index:05d}-of-{total_shards:05d}.mega"
        shard_path = dst_dir / shard_name
        shard_payload = _renumber_payload_offsets(shard_tensors)
        if compression_codec != MEGA_COMPRESSION_NONE:
            encoded_payload = dst_dir / f".{shard_name}.payload.tmp"
            try:
                shard_payload = list(
                    megacpp.encode_file_tensors(
                        str(encoded_payload),
                        shard_payload,
                        int(compression_codec),
                        int(compression_level),
                        float(compression_min_ratio),
                    )
                )
            except Exception:
                if encoded_payload.exists():
                    encoded_payload.unlink()
                raise
        base_meta = global_metadata if shard_metadata == "all" else minimal_metadata
        if shard_metadata not in {"first", "all", "minimal"}:
            raise ValueError("shard_metadata must be one of: first, all, minimal")
        if shard_metadata == "first" and shard_index == 1:
            base_meta = {
                **minimal_metadata,
                "general.name": global_metadata.get("general.name", ""),
                "general.basename": global_metadata.get("general.basename", ""),
            }
        if shard_metadata == "minimal":
            base_meta = minimal_metadata
        shard_meta = {
            **base_meta,
            "mega.shard.index": shard_index - 1,
            "mega.shard.count": total_shards,
            "mega.shard.payload.nbytes": sum(int(t["stored_nbytes"]) for t in shard_payload),
        }
        megacpp.write_file(str(shard_path), shard_payload, shard_meta, int(alignment))
        if compression_codec != MEGA_COMPRESSION_NONE and encoded_payload.exists():
            encoded_payload.unlink()
        shard_paths.append(shard_path)

    index_path = dst_dir / f"{basename}.mega.index.json"
    _write_index(
        index_path,
        src_dir,
        basename,
        shard_paths,
        shards,
        total_payload,
        max_bytes,
        global_metadata,
    )
    if signing is not None:
        from .signing import sign_artifact

        sign_artifact(index_path, signing)
    elapsed = time.perf_counter() - started
    return ConvertResult(index_path, shard_paths, len(tensors), total_payload, elapsed)


def resolve_artifacts(path: str | Path) -> list[str]:
    """Return concrete MEGA shard files for a shard path or MEGA JSON index."""
    src = Path(path)
    if src.name.endswith(".mega.index.json"):
        return _read_index(src)
    return [str(src)]


def build_tensor_table(model_dir: Path) -> tuple[list[dict[str, Any]], int]:
    tensors: list[dict[str, Any]] = []
    payload_offset = 0
    for shard in _safetensors_files(model_dir):
        data_start, header = _read_safetensors_header(shard)
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(spec, dict):
                raise ValueError(f"{shard}: tensor {name!r} spec must be an object")
            dtype = SAFETENSORS_DTYPE_TO_MEGA.get(str(spec["dtype"]))
            if dtype is None:
                raise ValueError(f"{shard}: unsupported safetensors dtype {spec['dtype']!r}")
            start, end = _data_offsets(shard, name, spec)
            stored_nbytes = end - start
            tensors.append(
                {
                    "name": name,
                    "shape": [int(dim) for dim in spec["shape"]],
                    "logical_dtype": dtype,
                    "storage_format": "raw_dense",
                    "payload_offset": payload_offset,
                    "logical_nbytes": stored_nbytes,
                    "stored_nbytes": stored_nbytes,
                    "src_filename": str(shard),
                    "src_offset": data_start + start,
                }
            )
            payload_offset += stored_nbytes
    return tensors, payload_offset


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be an object")
    return 8 + header_len, header


def _safetensors_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{model_dir}: no .safetensors files found")
    return files


def _data_offsets(path: Path, name: str, spec: Mapping[str, Any]) -> tuple[int, int]:
    data_offsets = spec.get("data_offsets")
    if (
        not isinstance(data_offsets, list)
        or len(data_offsets) != 2
        or int(data_offsets[1]) < int(data_offsets[0])
    ):
        raise ValueError(f"{path}: invalid data_offsets for tensor {name}")
    return int(data_offsets[0]), int(data_offsets[1])


def _parse_size(value: int | str) -> int:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("max_shard_size must be positive")
        return value
    text = str(value).strip().upper().replace("_", "")
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
    }
    for suffix, scale in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * scale)
    return int(text)


def _compression_codec(name: str) -> int:
    key = str(name).strip().lower()
    if key not in MEGA_COMPRESSION_BY_NAME:
        raise ValueError("compression must be one of: auto, none, zstd")
    return MEGA_COMPRESSION_BY_NAME[key]


def _plan_shards(tensors: Iterable[dict[str, Any]], max_shard_bytes: int) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for tensor in tensors:
        size = int(tensor["stored_nbytes"])
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(tensor)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def _renumber_payload_offsets(tensors: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    for tensor in tensors:
        item = dict(tensor)
        item["payload_offset"] = offset
        offset += int(item["stored_nbytes"])
        out.append(item)
    return out


def _minimal_shard_metadata(metadata: Mapping[str, Any], alignment: int) -> dict[str, Any]:
    return {
        "general.architecture": str(metadata.get("general.architecture", "unknown")),
        "general.alignment": int(alignment),
        "mega.artifact.kind": str(metadata.get("mega.artifact.kind", "model")),
        "mega.layout.tensor_order": str(metadata.get("mega.layout.tensor_order", "safetensors_header")),
        "mega.tensor_info.format": str(metadata.get("mega.tensor_info.format", "self_describing")),
    }


def _token_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def _token_id_from_tokenizer(tokenizer_json: dict[str, Any], token: str) -> int:
    model = tokenizer_json.get("model", {})
    vocab = model.get("vocab", {}) if isinstance(model, dict) else {}
    if isinstance(vocab, dict) and token in vocab:
        return int(vocab[token])
    for item in tokenizer_json.get("added_tokens", []):
        if isinstance(item, dict) and item.get("content") == token:
            return int(item.get("id", -1))
    return -1


def _tokenizer_metadata(model_dir: Path) -> dict[str, Any]:
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        return {}
    tokenizer_json = _read_json(tokenizer_path)
    tokenizer_config = (
        _read_json(model_dir / "tokenizer_config.json")
        if (model_dir / "tokenizer_config.json").exists()
        else {}
    )
    model = tokenizer_json.get("model", {})
    if not isinstance(model, dict):
        return {}
    vocab = model.get("vocab", {})
    if not isinstance(vocab, dict):
        return {}
    tokens = [""] * len(vocab)
    for token, token_id in vocab.items():
        tokens[int(token_id)] = str(token)
    merges = model.get("merges", [])
    if merges and not all(isinstance(item, str) for item in merges):
        merges = [" ".join(str(part) for part in item) for item in merges]

    eos_token = _token_text(tokenizer_config.get("eos_token", ""))
    bos_token = _token_text(tokenizer_config.get("bos_token", ""))
    unk_token = _token_text(tokenizer_config.get("unk_token", ""))
    pad_token = _token_text(tokenizer_config.get("pad_token", ""))
    model_type = str(model.get("type", "")).lower()
    return {
        "tokenizer.model": "bpe" if model_type == "bpe" else model_type,
        "tokenizer.tokens": tokens,
        "tokenizer.token_type": [0] * len(tokens),
        "tokenizer.merges": [str(item) for item in merges],
        "tokenizer.pre_tokenizer": "bytelevel",
        "tokenizer.bos_token": bos_token,
        "tokenizer.eos_token": eos_token,
        "tokenizer.unknown_token": unk_token,
        "tokenizer.padding_token": pad_token,
        "tokenizer.bos_token_id": _token_id_from_tokenizer(tokenizer_json, bos_token)
        if bos_token
        else -1,
        "tokenizer.eos_token_id": _token_id_from_tokenizer(tokenizer_json, eos_token)
        if eos_token
        else -1,
        "tokenizer.unknown_token_id": _token_id_from_tokenizer(tokenizer_json, unk_token)
        if unk_token
        else -1,
        "tokenizer.padding_token_id": _token_id_from_tokenizer(tokenizer_json, pad_token)
        if pad_token
        else -1,
        "tokenizer.add_bos_token": bool(tokenizer_config.get("add_bos_token", False)),
        "tokenizer.add_eos_token": bool(tokenizer_config.get("add_eos_token", False)),
    }


def _model_metadata(model_dir: Path, tensor_count: int, total_payload: int, alignment: int) -> dict[str, Any]:
    config = _read_json(model_dir / "config.json")
    text_config = config.get("text_config", {})
    if not isinstance(text_config, dict):
        text_config = {}
    architecture = str(config.get("model_type") or text_config.get("model_type") or "unknown")
    meta: dict[str, Any] = {
        "general.architecture": architecture,
        "general.name": model_dir.name,
        "general.basename": model_dir.name,
        "general.alignment": int(alignment),
        "mega.artifact.kind": "model",
        "mega.layout.tensor_order": "safetensors_header",
        "mega.tensor_info.format": "self_describing",
        "mega.source.format": "safetensors",
        "mega.source.path": str(model_dir),
        "mega.tensor.count": int(tensor_count),
        "mega.payload.nbytes": int(total_payload),
    }
    archs = config.get("architectures")
    if isinstance(archs, list):
        meta["model.architectures"] = [str(item) for item in archs]
    for key, mega_key in [
        ("max_position_embeddings", "context_length"),
        ("hidden_size", "embedding_length"),
        ("num_hidden_layers", "block_count"),
        ("intermediate_size", "feed_forward_length"),
        ("num_attention_heads", "attention.head_count"),
        ("num_key_value_heads", "attention.head_count_kv"),
        ("vocab_size", "vocab_size"),
    ]:
        if key in text_config:
            meta[f"{architecture}.{mega_key}"] = int(text_config[key])
    rope = text_config.get("rope_parameters", {})
    if isinstance(rope, dict) and "rope_dim" in rope:
        meta[f"{architecture}.rope.dimension_count"] = int(rope["rope_dim"])
    elif "head_dim" in text_config:
        meta[f"{architecture}.rope.dimension_count"] = int(text_config["head_dim"])
    return meta


def _write_index(
    path: Path,
    source_dir: Path,
    basename: str,
    shard_paths: list[Path],
    shards: list[list[dict[str, Any]]],
    total_payload: int,
    max_shard_bytes: int,
    metadata: Mapping[str, Any],
) -> None:
    weight_map = {}
    shard_table = {}
    for shard_path, shard_tensors in zip(shard_paths, shards):
        shard_name = shard_path.name
        for tensor in shard_tensors:
            weight_map[str(tensor["name"])] = shard_name
    for shard_path in shard_paths:
        shard_table[shard_path.name] = str(shard_path)
    payload = {
        "metadata": {
            **dict(metadata),
            "format": "mega-sharded-index-v1",
            "source": str(source_dir),
            "basename": basename,
            "total_size": int(total_payload),
            "max_shard_size": int(max_shard_bytes),
        },
        "weight_map": weight_map,
        "shards": shard_table,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def read_index_metadata(path: str | Path) -> dict[str, Any]:
    data = _read_index_data(Path(path))
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return dict(metadata)


def _read_index_data(path: Path) -> dict[str, Any]:
    if not path.name.endswith(".mega.index.json"):
        raise ValueError(f"{path}: MEGA index must be .mega.index.json")
    return _read_json(path)


def _read_index(path: Path) -> list[str]:
    data = _read_index_data(path)
    shards = data.get("shards")
    if not isinstance(shards, dict):
        raise ValueError(f"{path}: missing shards table")
    out = []
    for shard_value in shards.values():
        shard = Path(str(shard_value))
        if not shard.is_absolute():
            shard = path.parent / shard
        out.append(str(shard))
    return sorted(set(out))
