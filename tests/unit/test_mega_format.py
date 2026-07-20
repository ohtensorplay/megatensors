# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import os
import shutil
import struct
import subprocess
import zlib
import zstandard as zstd

import pytest

from megatensors import cpp as megacpp
from megatensors import (
    MegaKvMetadata,
    MegaTensorsMetadata,
    append_footer_overlay,
    iter_tensors,
    load_kv_tensor,
    load_model,
    load_state_dict,
    load_tensor,
    load_tokenizer,
    mega_open,
    open_kv_cache,
    write_kv_cache,
)
from megatensors.frameworks import get_framework_op
from megatensors.loader import MegaTensorsFileLoader
from megatensors.st_types import Device


MEGA_MAGIC = 0x4147454D
MEGAKV_MAGIC = 0x564B474D
MEGA_VERSION = 1
MEGAKV_VERSION = 1
META_ARRAY = 12
META_UINT32 = 4
META_STRING = 11
TENSOR_FLAG_COMPRESSED = 1
TENSOR_FLAG_BYTE_SHUFFLED = 2
COMPRESSION_NONE = 0
COMPRESSION_ZSTD = 1
CHECKSUM_NONE = 0
CHECKSUM_CRC32 = 1
CHECKSUM_SHA256 = 2
TENSOR_DIR_HAS_STORAGE_FORMAT = 1 << 0
TENSOR_DIR_HAS_STORED_NBYTES = 1 << 1
TENSOR_DIR_HAS_TENSOR_FLAGS = 1 << 2
TENSOR_DIR_HAS_COMPRESSION_CODEC = 1 << 3
TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE = 1 << 4
TENSOR_DIR_HAS_CHECKSUM = 1 << 5
NO_SEGMENT_ID = 0xFFFFFFFF
MOE_ROLE_GATE = 1
MOE_ROLE_UP = 2
MOE_ROLE_DOWN = 3


def _pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_compact_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _dtype_nbytes(dtype: str) -> int:
    if dtype in {"BOOL", "I8", "U8", "F8_E5M2", "F8_E4M3", "F8_E8M0"}:
        return 1
    if dtype in {"I16", "U16", "F16", "BF16"}:
        return 2
    if dtype in {"I32", "U32", "F32"}:
        return 4
    if dtype in {"I64", "U64", "F64"}:
        return 8
    raise ValueError(dtype)


def _logical_nbytes(dtype: str, shape: list[int]) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    if dtype == "F4":
        return (elements + 1) // 2
    return elements * _dtype_nbytes(dtype)


def _pack_meta_value(value):
    if isinstance(value, str):
        return struct.pack("<I", META_STRING) + _pack_string(value)
    if isinstance(value, int):
        return struct.pack("<I", META_UINT32) + struct.pack("<I", value)
    if isinstance(value, list):
        if not value:
            return struct.pack("<IIQ", META_ARRAY, META_STRING, 0)
        if all(isinstance(item, str) for item in value):
            payload = struct.pack("<IIQ", META_ARRAY, META_STRING, len(value))
            return payload + b"".join(_pack_string(item) for item in value)
        if all(isinstance(item, int) for item in value):
            payload = struct.pack("<IIQ", META_ARRAY, META_UINT32, len(value))
            return payload + b"".join(struct.pack("<I", item) for item in value)
    raise TypeError(value)


def _byte_shuffle(payload: bytes, elem_size: int) -> bytes:
    assert len(payload) % elem_size == 0
    return b"".join(payload[i::elem_size] for i in range(elem_size))


def _crc32_checksum(payload: bytes) -> bytes:
    return struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF) + b"\0" * 28


def _sha256_checksum(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _run_openssl(tmp_path, *args):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl CLI is required for trust-chain tests")
    subprocess.run(
        [openssl, *args],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _trust_statement(
    payload_sha256: str,
    *,
    publisher: str = "CN=MEGA Test Publisher",
    model_id: str = "test-model",
    created_at: int = 1_700_000_000,
    expires_at: int = 4_102_444_800,
) -> bytes:
    return (
        "MEGA-TRUST-v1\n"
        "format_version=1\n"
        "artifact_kind=model\n"
        f"publisher={publisher}\n"
        f"model_id={model_id}\n"
        f"payload_sha256={payload_sha256}\n"
        f"created_at={created_at}\n"
        f"expires_at={expires_at}\n"
    ).encode("utf-8")


def _make_signed_trust_statement(tmp_path, statement: bytes, *, code_signing=True):
    root_cnf = os.path.join(tmp_path, "root.cnf")
    leaf_ext = os.path.join(tmp_path, "leaf.ext")
    statement_path = os.path.join(tmp_path, "statement.txt")
    signature_path = os.path.join(tmp_path, "statement.sig")
    with open(root_cnf, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "[req]",
                    "prompt = no",
                    "distinguished_name = dn",
                    "x509_extensions = v3_ca",
                    "[dn]",
                    "CN = MEGA Test Root CA",
                    "[v3_ca]",
                    "basicConstraints = critical,CA:true",
                    "keyUsage = critical,keyCertSign,cRLSign",
                    "subjectKeyIdentifier = hash",
                    "",
                ]
            )
        )
    leaf_lines = [
        "[v3_leaf]",
        "basicConstraints = critical,CA:false",
        "keyUsage = critical,digitalSignature",
    ]
    if code_signing:
        leaf_lines.append("extendedKeyUsage = codeSigning")
    leaf_lines.extend(
        [
            "subjectKeyIdentifier = hash",
            "authorityKeyIdentifier = keyid,issuer",
            "",
        ]
    )
    with open(leaf_ext, "w", encoding="utf-8") as f:
        f.write("\n".join(leaf_lines))
    _run_openssl(
        tmp_path,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-sha256",
        "-days",
        "30",
        "-nodes",
        "-config",
        root_cnf,
        "-keyout",
        "root.key",
        "-out",
        "root.pem",
    )
    _run_openssl(
        tmp_path,
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-subj",
        "/CN=MEGA Test Publisher",
        "-keyout",
        "leaf.key",
        "-out",
        "leaf.csr",
    )
    _run_openssl(
        tmp_path,
        "x509",
        "-req",
        "-in",
        "leaf.csr",
        "-CA",
        "root.pem",
        "-CAkey",
        "root.key",
        "-CAcreateserial",
        "-days",
        "30",
        "-sha256",
        "-extfile",
        leaf_ext,
        "-extensions",
        "v3_leaf",
        "-out",
        "leaf.pem",
    )
    with open(statement_path, "wb") as f:
        f.write(statement)
    _run_openssl(
        tmp_path,
        "dgst",
        "-sha256",
        "-sign",
        "leaf.key",
        "-out",
        signature_path,
        statement_path,
    )
    with open(os.path.join(tmp_path, "root.pem"), encoding="utf-8") as f:
        root_pem = f.read()
    with open(os.path.join(tmp_path, "leaf.pem"), encoding="utf-8") as f:
        leaf_pem = f.read()
    with open(signature_path, "rb") as f:
        signature_b64 = base64.b64encode(f.read()).decode("ascii")
    return root_pem, leaf_pem, signature_b64


def _write_mega(
    path,
    tensors,
    metadata=None,
    alignment=32,
    segments=None,
    moe_experts=None,
):
    segments = segments or []
    moe_experts = moe_experts or []
    has_chunks = any(tensor.get("chunks") for tensor in tensors)
    metadata = {
        "general.architecture": "test",
        "general.alignment": alignment,
        "mega.layout.tensor_order": "original",
        "mega.tensor_info.format": "self_describing",
        **(metadata or {}),
    }
    chunk_records = []
    if has_chunks:
        metadata.setdefault("mega.chunk_directory.format", "v1")
        metadata["mega.chunk_directory.count"] = sum(
            len(tensor.get("chunks", [])) for tensor in tensors
        )
    if segments:
        metadata.setdefault("mega.segment_directory.format", "v1")
        metadata["mega.segment_directory.count"] = len(segments)
    if moe_experts:
        metadata.setdefault("mega.moe.expert_table.format", "v1")
        metadata["mega.moe.expert_table.count"] = len(moe_experts)
    header = bytearray()
    header += struct.pack("<IIQQ", MEGA_MAGIC, MEGA_VERSION, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _pack_string(key)
        header += _pack_meta_value(value)

    payload = bytearray()
    for tensor_id, tensor in enumerate(tensors):
        chunks = tensor.get("chunks", [])
        tensor_payload = b""
        if chunks:
            payload_offset = tensor.get("payload_offset", 0)
            stored_nbytes = tensor.get("stored_nbytes", 0)
            for chunk_id, chunk in enumerate(chunks):
                chunk_payload_offset = len(payload)
                chunk_data = chunk["data"]
                payload += chunk_data
                chunk_records.append(
                    {
                        "tensor_id": tensor_id,
                        "chunk_id": chunk.get("chunk_id", chunk_id),
                        "logical_offset": chunk["logical_offset"],
                        "logical_size": chunk["logical_size"],
                        "payload_offset": chunk_payload_offset,
                        "stored_size": len(chunk_data),
                        "codec": chunk.get("codec", COMPRESSION_NONE),
                        "flags": chunk.get("flags", 0),
                        "checksum_type": chunk.get("checksum_type", CHECKSUM_NONE),
                        "checksum": chunk.get("checksum"),
                    }
                )
        else:
            payload_offset = len(payload)
            payload += tensor["data"]
            stored_nbytes = tensor.get("stored_nbytes", len(tensor["data"]))
            tensor_payload = tensor["data"]
        tensor_checksum_type = tensor.get("checksum_type", CHECKSUM_NONE)
        tensor_checksum = tensor.get("checksum")
        if tensor_checksum is None:
            if tensor_checksum_type == CHECKSUM_CRC32:
                tensor_checksum = _crc32_checksum(tensor_payload if not chunks else b"")
            elif tensor_checksum_type == CHECKSUM_SHA256:
                tensor_checksum = _sha256_checksum(tensor_payload if not chunks else b"")
            else:
                tensor_checksum = b"\0" * 32
        assert len(tensor_checksum) == 32
        logical_nbytes = tensor.get(
            "logical_nbytes",
            _logical_nbytes(tensor["logical_dtype"], tensor["shape"]),
        )
        assert logical_nbytes == _logical_nbytes(tensor["logical_dtype"], tensor["shape"])
        storage_format = tensor.get("storage_format", "raw_dense")
        tensor_flags = tensor.get("tensor_flags", 0)
        compression_codec = tensor.get("compression_codec", 0)
        shuffle_elem_size = tensor.get("shuffle_elem_size", 0)
        dir_flags = 0
        if storage_format != "raw_dense":
            dir_flags |= TENSOR_DIR_HAS_STORAGE_FORMAT
        if stored_nbytes != logical_nbytes:
            dir_flags |= TENSOR_DIR_HAS_STORED_NBYTES
        if tensor_flags != 0:
            dir_flags |= TENSOR_DIR_HAS_TENSOR_FLAGS
        if compression_codec != COMPRESSION_NONE:
            dir_flags |= TENSOR_DIR_HAS_COMPRESSION_CODEC
        if shuffle_elem_size != 0:
            dir_flags |= TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE
        if tensor_checksum_type != CHECKSUM_NONE:
            dir_flags |= TENSOR_DIR_HAS_CHECKSUM

        dims_reversed = list(reversed(tensor["shape"]))
        header += _pack_compact_string(tensor["name"])
        header += struct.pack("<I", dir_flags)
        header += struct.pack("<I", len(dims_reversed))
        for dim in dims_reversed:
            header += struct.pack("<Q", dim)
        header += _pack_compact_string(tensor["logical_dtype"])
        header += struct.pack("<Q", payload_offset)
        if dir_flags & TENSOR_DIR_HAS_STORAGE_FORMAT:
            header += _pack_compact_string(storage_format)
        if dir_flags & TENSOR_DIR_HAS_STORED_NBYTES:
            header += struct.pack("<Q", stored_nbytes)
        if dir_flags & TENSOR_DIR_HAS_TENSOR_FLAGS:
            header += struct.pack("<I", tensor_flags)
        if dir_flags & TENSOR_DIR_HAS_COMPRESSION_CODEC:
            header += struct.pack("<I", compression_codec)
        if dir_flags & TENSOR_DIR_HAS_SHUFFLE_ELEM_SIZE:
            header += struct.pack("<I", shuffle_elem_size)
        if dir_flags & TENSOR_DIR_HAS_CHECKSUM:
            header += struct.pack("<I32s", tensor_checksum_type, tensor_checksum)

    for chunk in chunk_records:
        checksum = chunk["checksum"]
        if checksum is None:
            if chunk["checksum_type"] == CHECKSUM_CRC32:
                checksum = _crc32_checksum(
                    payload[chunk["payload_offset"] : chunk["payload_offset"] + chunk["stored_size"]]
                )
            elif chunk["checksum_type"] == CHECKSUM_SHA256:
                checksum = _sha256_checksum(
                    payload[chunk["payload_offset"] : chunk["payload_offset"] + chunk["stored_size"]]
                )
            else:
                checksum = b"\0" * 32
        assert len(checksum) == 32
        header += struct.pack(
            "<IIQQQQIIII32s",
            chunk["tensor_id"],
            chunk["chunk_id"],
            chunk["logical_offset"],
            chunk["logical_size"],
            chunk["payload_offset"],
            chunk["stored_size"],
            chunk["codec"],
            chunk["flags"],
            chunk["checksum_type"],
            0,
            checksum,
        )

    auto_payload_size = len(payload)
    for segment_id, segment in enumerate(segments):
        payload_offset = segment.get("payload_offset", 0)
        payload_size = segment.get("payload_size", 0)
        if payload_offset == "auto":
            payload_offset = 0
        if payload_size == "auto":
            payload_size = auto_payload_size
        header += struct.pack(
            "<IIIIIIIIQQIIIIII",
            segment.get("segment_id", segment_id),
            segment.get("kind", 0),
            segment.get("priority", 0),
            segment.get("flags", 0),
            segment.get("layer_start", 0),
            segment.get("layer_end", 0),
            segment.get("expert_start", 0),
            segment.get("expert_end", 0),
            payload_offset,
            payload_size,
            segment.get("first_tensor_id", 0),
            segment.get("tensor_count", 0),
            segment.get("prefetch_group", 0),
            segment.get("device_hint", 0),
            segment.get("cache_policy", 0),
            0,
        )

    for expert in moe_experts:
        header += struct.pack(
            "<IIIIIIII",
            expert["tensor_id"],
            expert["layer_id"],
            expert["expert_id"],
            expert["expert_role"],
            expert.get("segment_id", NO_SEGMENT_ID),
            expert.get("prefetch_group", 0),
            expert.get("cache_policy", 0),
            0,
        )

    pad = (-len(header)) % alignment
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\0" * pad)
        f.write(payload)


def test_cpp_write_file_streams_tensor_payloads(tmp_path):
    src = tmp_path / "source.bin"
    dst = tmp_path / "streamed.mega"
    src.write_bytes(b"xxxx" + b"\x01\x02\x03\x04" + b"yyyy" + b"\x05\x06")

    megacpp.write_file(
        str(dst),
        [
            {
                "name": "a",
                "shape": [2],
                "logical_dtype": "U16",
                "storage_format": "raw_dense",
                "payload_offset": 0,
                "logical_nbytes": 4,
                "stored_nbytes": 4,
                "src_filename": str(src),
                "src_offset": 4,
            },
            {
                "name": "b",
                "shape": [2],
                "logical_dtype": "U8",
                "storage_format": "raw_dense",
                "payload_offset": 4,
                "logical_nbytes": 2,
                "stored_nbytes": 2,
                "src_filename": str(src),
                "src_offset": 12,
            },
        ],
        {
            "general.architecture": "test",
            "general.alignment": 32,
            "mega.layout.tensor_order": "original",
            "mega.tensor_info.format": "self_describing",
        },
        32,
    )

    fw = get_framework_op("pt")
    meta = MegaTensorsMetadata.from_file(str(dst), fw)
    assert list(meta.tensors) == ["a", "b"]
    with open(dst, "rb") as f:
        f.seek(meta.header_length)
        assert f.read(6) == b"\x01\x02\x03\x04\x05\x06"


def _write_megakv(path, entries, metadata=None, alignment=32):
    metadata = {
        "general.alignment": alignment,
        "megakv.format": "v1",
        **(metadata or {}),
    }
    header = bytearray()
    header += struct.pack("<IIQQ", MEGAKV_MAGIC, MEGAKV_VERSION, len(entries), len(metadata))
    for key, value in metadata.items():
        header += _pack_string(key)
        header += _pack_meta_value(value)

    payload = bytearray()
    for entry_id, entry in enumerate(entries):
        payload_offset = len(payload)
        payload += entry["data"]
        checksum_type = entry.get("checksum_type", CHECKSUM_NONE)
        checksum = entry.get("checksum")
        if checksum is None:
            if checksum_type == CHECKSUM_CRC32:
                checksum = _crc32_checksum(entry["data"])
            elif checksum_type == CHECKSUM_SHA256:
                checksum = _sha256_checksum(entry["data"])
            else:
                checksum = b"\0" * 32
        assert len(checksum) == 32
        header += _pack_string(entry.get("name", f"kv.{entry_id}"))
        header += struct.pack(
            "<IIQQQI",
            entry.get("layer_id", 0),
            entry.get("kv_role", 0),
            entry.get("sequence_id", 0),
            entry.get("token_start", 0),
            entry.get("token_count", 0),
            len(entry["shape"]),
        )
        for dim in entry["shape"]:
            header += struct.pack("<Q", dim)
        header += _pack_string(entry["logical_dtype"])
        header += struct.pack(
            "<QQQIIII",
            payload_offset,
            entry["logical_nbytes"],
            entry.get("stored_nbytes", len(entry["data"])),
            entry.get("flags", 0),
            entry.get("codec", COMPRESSION_NONE),
            entry.get("shuffle_elem_size", 0),
            0,
        )
        header += struct.pack("<I32s", checksum_type, checksum)

    pad = (-len(header)) % alignment
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\0" * pad)
        f.write(payload)


def test_mega_metadata_and_raw_dense_load(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "raw.mega")
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 3],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": 24,
                "data": struct.pack("<6f", *values),
            }
        ],
        metadata={
            "model.class": "torch.nn.Linear",
            "model.init.in_features": 2,
            "model.init.out_features": 1,
            "model.state_dict.prefix": "linear.",
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    assert meta.metadata["mega.tensor_info.format"] == "self_describing"
    frame = meta.tensors["layers.0.weight"]
    assert frame.shape == [2, 3]
    assert frame.storage_format == "raw_dense"
    assert frame.logical_nbytes == 24
    assert frame.stored_nbytes == 24

    expected = torch.tensor(values, dtype=torch.float32).reshape(2, 3)
    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        actual = f.get_tensor("layers.0.weight").clone()
    assert torch.equal(actual, expected)


def test_mega_embeds_model_info_and_tokenizer_metadata(tmp_path, framework):
    path = os.path.join(tmp_path, "model-info.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    _write_mega(
        path,
        [
            {
                "name": "token_embd.weight",
                "shape": [1, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "general.architecture": "qwen3",
            "general.name": "Qwen3 MEGA",
            "general.basename": "qwen3",
            "general.size_label": "0.6B",
            "qwen3.context_length": 32768,
            "qwen3.embedding_length": 1024,
            "qwen3.block_count": 28,
            "qwen3.feed_forward_length": 3072,
            "qwen3.attention.head_count": 16,
            "qwen3.attention.head_count_kv": 8,
            "qwen3.rope.dimension_count": 64,
            "qwen3.vocab_size": 4,
            "tokenizer.model": "wordlevel",
            "tokenizer.tokens": ["<unk>", "<s>", "</s>", "hello"],
            "tokenizer.token_type": [2, 3, 3, 1],
            "tokenizer.bos_token_id": 1,
            "tokenizer.eos_token_id": 2,
            "tokenizer.unknown_token_id": 0,
            "tokenizer.bos_token": "<s>",
            "tokenizer.eos_token": "</s>",
            "tokenizer.unknown_token": "<unk>",
            "tokenizer.pre_tokenizer": "whitespace",
            "tokenizer.add_bos_token": 1,
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    model = meta.get_model_info()
    assert model.architecture == "qwen3"
    assert model.context_length == 32768
    assert model.attention_head_count_kv == 8
    tokenizer = meta.get_tokenizer_info()
    assert tokenizer.model == "wordlevel"
    assert tokenizer.tokens[-1] == "hello"
    assert tokenizer.token_types == [2, 3, 3, 1]
    assert tokenizer.bos_token_id == 1
    assert tokenizer.add_bos_token is True
    hf = meta.get_hf_tokenizer()
    assert hf.encode("hello", add_special_tokens=False) == [3]
    assert hf.unk_token == "<unk>"
    hf2 = load_tokenizer(path)
    assert hf2.encode("hello", add_special_tokens=False) == [3]
    index_path = os.path.join(tmp_path, "model.mega.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": dict(meta.metadata),
                "weight_map": {"token_embd.weight": os.path.basename(path)},
                "shards": {os.path.basename(path): path},
            },
            f,
        )
    hf3 = load_tokenizer(index_path)
    assert hf3.encode("hello", add_special_tokens=False) == [3]


def test_mega_load_state_dict_and_torch_model_api(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch module assertions")
    import torch

    path = os.path.join(tmp_path, "torch-api.mega")
    weight = struct.pack("<2f", 2.0, 3.0)
    bias = struct.pack("<f", 0.5)
    _write_mega(
        path,
        [
            {
                "name": "linear.weight",
                "shape": [1, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(weight),
                "data": weight,
            },
            {
                "name": "linear.bias",
                "shape": [1],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(bias),
                "data": bias,
            },
        ],
        metadata={
            "model.class": "torch.nn.Linear",
            "model.init.in_features": 2,
            "model.init.out_features": 1,
            "model.state_dict.prefix": "linear.",
        },
    )

    state_dict = load_state_dict(
        path,
        device="cpu",
        nogds=True,
        key_mapping=lambda key: key.removeprefix("linear."),
    )
    assert list(state_dict.keys()) == ["weight", "bias"]
    assert torch.equal(load_tensor(path, "linear.bias", nogds=True), torch.tensor([0.5]))
    assert [name for name, _ in iter_tensors(path, nogds=True)] == [
        "linear.weight",
        "linear.bias",
    ]
    model = load_model(
        path,
        device="cpu",
        nogds=True,
    )
    actual = model(torch.tensor([[4.0, 5.0]], dtype=torch.float32))
    assert torch.equal(actual, torch.tensor([[23.5]], dtype=torch.float32))

    constructor_calls = []

    def linear_factory(**kwargs):
        constructor_calls.append(kwargs)
        return torch.nn.Linear(**kwargs)

    factory_model = load_model(
        path,
        model_class=linear_factory,
        device="cpu",
        nogds=True,
    )
    assert constructor_calls == [{"in_features": 2, "out_features": 1}]
    assert torch.equal(
        factory_model(torch.tensor([[4.0, 5.0]], dtype=torch.float32)),
        torch.tensor([[23.5]], dtype=torch.float32),
    )


def test_megakv_sidecar_loads_raw_kv_entry(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "cache.megakv")
    values = [0.5, 1.5, 2.5, 3.5]
    payload = struct.pack("<4f", *values)
    _write_megakv(
        path,
        [
            {
                "name": "layers.0.key",
                "layer_id": 0,
                "kv_role": 1,
                "sequence_id": 42,
                "token_start": 8,
                "token_count": 2,
                "shape": [1, 2, 2],
                "logical_dtype": "F32",
                "logical_nbytes": len(payload),
                "checksum_type": CHECKSUM_CRC32,
                "data": payload,
            }
        ],
        metadata={"general.architecture": "qwen3"},
    )

    kv = open_kv_cache(path)
    assert kv.metadata["megakv.format"] == "v1"
    assert kv.select_entries(layer_id=0, kv_role=1)[0].sequence_id == 42
    actual = kv.load_entry("layers.0.key", Device.from_str("cpu")).get_raw().clone()
    actual2 = load_kv_tensor(path, "layers.0.key").clone()
    expected = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 2)
    assert torch.equal(actual, expected)
    assert torch.equal(actual2, expected)


def test_write_kv_cache_writes_sha256_checked_sidecar(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "written-cache.megakv")
    values = [1.0, 3.0, 5.0, 7.0]
    payload = struct.pack("<4f", *values)
    write_kv_cache(
        path,
        [
            {
                "name": "kv.layer.0.k",
                "layer_id": 0,
                "kv_role": 1,
                "sequence_id": 9,
                "token_start": 0,
                "token_count": 2,
                "shape": [1, 2, 2],
                "logical_dtype": "F32",
                "data": payload,
            }
        ],
        metadata={
            "mega.kv.model_payload_hash": "payload-hash",
            "mega.kv.tokenizer_hash": "tokenizer-hash",
        },
    )

    kv = open_kv_cache(path)
    entry = kv.entries_by_name["kv.layer.0.k"]
    assert entry.checksum_type == CHECKSUM_SHA256
    assert entry.checksum == hashlib.sha256(payload).digest()
    actual = load_kv_tensor(path, "kv.layer.0.k").clone()
    expected = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 2)
    assert torch.equal(actual, expected)


def test_megakv_sidecar_rejects_bad_crc32(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-cache.megakv")
    payload = struct.pack("<2f", 1.0, 2.0)
    _write_megakv(
        path,
        [
            {
                "name": "layers.0.value",
                "layer_id": 0,
                "kv_role": 2,
                "sequence_id": 1,
                "token_start": 0,
                "token_count": 1,
                "shape": [1, 2],
                "logical_dtype": "F32",
                "logical_nbytes": len(payload),
                "checksum_type": CHECKSUM_CRC32,
                "checksum": struct.pack("<I", 0xDEADBEEF) + b"\0" * 28,
                "data": payload,
            }
        ],
    )

    kv = MegaKvMetadata.from_file(path, framework)
    with pytest.raises(Exception, match="checksum mismatch"):
        kv.load_entry("layers.0.value", Device.from_str("cpu"))


def test_mega_payload_crc32_artifact_hash_is_explicit(tmp_path, framework):
    path = os.path.join(tmp_path, "payload-hash.mega")
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    expected_crc = zlib.crc32(payload) & 0xFFFFFFFF
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={"mega.hash.payload.crc32": expected_crc},
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    assert meta.compute_payload_crc32() == expected_crc
    assert meta.verify_payload_crc32()
    with pytest.raises(ValueError, match="payload CRC32 mismatch"):
        meta.verify_payload_crc32(0xDEADBEEF)


def test_mega_payload_sha256_artifact_hash_is_explicit(tmp_path, framework):
    path = os.path.join(tmp_path, "payload-sha256.mega")
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={"mega.hash.payload.sha256": expected_sha256},
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    assert meta.compute_payload_sha256() == expected_sha256
    assert meta.verify_payload_sha256()
    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        meta.verify_payload_sha256("00" * 32)


def test_mega_footer_overlay_updates_metadata_without_payload_rewrite(
    tmp_path, framework
):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "overlay.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    base_sha256 = hashlib.sha256(payload).hexdigest()
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "mega.hash.payload.sha256": "0" * 64,
            "mega.source.model_id": "base",
        },
    )
    base_size = os.path.getsize(path)
    append_footer_overlay(
        path,
        {
            "mega.hash.payload.sha256": base_sha256,
            "mega.source.model_id": "overlay",
        },
        generation=7,
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    assert meta.file_size_bytes > base_size
    assert meta.size_bytes == base_size
    assert meta.footer_overlay["present"] is True
    assert meta.footer_overlay["generation"] == 7
    assert meta.metadata["mega.source.model_id"] == "overlay"
    assert meta.compute_payload_sha256() == base_sha256
    assert meta.verify_payload_sha256()

    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        actual = f.get_tensor("layers.0.weight").clone()
    assert torch.equal(actual, torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_mega_footer_overlay_rejects_structural_override(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-overlay.mega")
    payload = struct.pack("<f", 1.0)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [1],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
    )
    append_footer_overlay(path, {"mega.chunk_directory.count": 1})

    with pytest.raises(Exception, match="cannot override structural key"):
        MegaTensorsMetadata.from_file(path, framework)


def test_mega_x509_trust_requires_external_authority_and_signed_payload(
    tmp_path, framework
):
    path = os.path.join(tmp_path, "trusted.mega")
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    statement = _trust_statement(payload_sha256)
    root_pem, leaf_pem, signature_b64 = _make_signed_trust_statement(
        tmp_path, statement
    )
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "mega.hash.payload.sha256": payload_sha256,
            "mega.trust.certificate.format": "x509-pem-v1",
            "mega.trust.certificate.leaf_pem": leaf_pem,
            "mega.trust.signature.algorithm": "sha256-rsa-pkcs1",
            "mega.trust.signature.statement": statement.decode("utf-8"),
            "mega.trust.signature.value": signature_b64,
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    trusted = meta.verify_trust(
        root_pem,
        warn=False,
        allowed_model_ids={"test-model"},
    )
    assert trusted["trusted"] is True
    assert trusted["risk"] == "trusted"
    assert trusted["chain_trusted"] is True
    assert trusted["certificate_policy_valid"] is True
    assert trusted["signature_valid"] is True
    assert trusted["payload_sha256"] == payload_sha256
    assert trusted["statement"]["publisher"] == "CN=MEGA Test Publisher"

    with pytest.warns(RuntimeWarning, match="source risk: untrusted_issuer"):
        untrusted = meta.verify_trust("", warn=True)
    assert untrusted["trusted"] is False
    assert untrusted["source_risk"] is True
    with pytest.raises(ValueError, match="source risk: untrusted_issuer"):
        meta.verify_trust("", strict=True, warn=False)


def test_mega_x509_trust_rejects_statement_not_bound_to_payload(
    tmp_path, framework
):
    path = os.path.join(tmp_path, "bad-trust-hash.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    statement = _trust_statement("00" * 32)
    root_pem, leaf_pem, signature_b64 = _make_signed_trust_statement(
        tmp_path, statement
    )
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "mega.hash.payload.sha256": payload_sha256,
            "mega.trust.certificate.format": "x509-pem-v1",
            "mega.trust.certificate.leaf_pem": leaf_pem,
            "mega.trust.signature.algorithm": "sha256-rsa-pkcs1",
            "mega.trust.signature.statement": statement.decode("utf-8"),
            "mega.trust.signature.value": signature_b64,
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    with pytest.warns(RuntimeWarning, match="source risk: hash_mismatch"):
        result = meta.verify_trust(root_pem)
    assert result["trusted"] is False
    assert result["risk"] == "hash_mismatch"
    assert result["payload_sha256"] == payload_sha256


def test_mega_x509_trust_rejects_publisher_certificate_mismatch(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-publisher.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    statement = _trust_statement(payload_sha256, publisher="unknown-publisher")
    root_pem, leaf_pem, signature_b64 = _make_signed_trust_statement(
        tmp_path, statement
    )
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "mega.hash.payload.sha256": payload_sha256,
            "mega.trust.certificate.format": "x509-pem-v1",
            "mega.trust.certificate.leaf_pem": leaf_pem,
            "mega.trust.signature.algorithm": "sha256-rsa-pkcs1",
            "mega.trust.signature.statement": statement.decode("utf-8"),
            "mega.trust.signature.value": signature_b64,
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    with pytest.warns(RuntimeWarning, match="source risk: publisher_certificate_mismatch"):
        result = meta.verify_trust(root_pem)
    assert result["trusted"] is False
    assert result["risk"] == "publisher_certificate_mismatch"


def test_mega_x509_trust_rejects_expired_statement(tmp_path, framework):
    path = os.path.join(tmp_path, "expired-statement.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    statement = _trust_statement(
        payload_sha256,
        created_at=100,
        expires_at=200,
    )
    root_pem, leaf_pem, signature_b64 = _make_signed_trust_statement(
        tmp_path, statement
    )
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "mega.hash.payload.sha256": payload_sha256,
            "mega.trust.certificate.format": "x509-pem-v1",
            "mega.trust.certificate.leaf_pem": leaf_pem,
            "mega.trust.signature.algorithm": "sha256-rsa-pkcs1",
            "mega.trust.signature.statement": statement.decode("utf-8"),
            "mega.trust.signature.value": signature_b64,
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    with pytest.warns(RuntimeWarning, match="source risk: statement_expired"):
        result = meta.verify_trust(root_pem, now=201)
    assert result["trusted"] is False
    assert result["risk"] == "statement_expired"


def test_mega_x509_trust_rejects_leaf_without_code_signing_usage(
    tmp_path, framework
):
    path = os.path.join(tmp_path, "bad-cert-policy.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    statement = _trust_statement(payload_sha256)
    root_pem, leaf_pem, signature_b64 = _make_signed_trust_statement(
        tmp_path, statement, code_signing=False
    )
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "mega.hash.payload.sha256": payload_sha256,
            "mega.trust.certificate.format": "x509-pem-v1",
            "mega.trust.certificate.leaf_pem": leaf_pem,
            "mega.trust.signature.algorithm": "sha256-rsa-pkcs1",
            "mega.trust.signature.statement": statement.decode("utf-8"),
            "mega.trust.signature.value": signature_b64,
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    with pytest.warns(RuntimeWarning, match="source risk: certificate_policy_invalid"):
        result = meta.verify_trust(root_pem)
    assert result["trusted"] is False
    assert result["risk"] == "certificate_policy_invalid"


def test_mega_x509_trust_rejects_noncanonical_statement(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-statement.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    statement = (
        "MEGA-TRUST-v1\n"
        f"payload_sha256={payload_sha256}\n"
        "artifact_kind=model\n"
    ).encode("utf-8")
    root_pem, leaf_pem, signature_b64 = _make_signed_trust_statement(
        tmp_path, statement
    )
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={
            "mega.hash.payload.sha256": payload_sha256,
            "mega.trust.certificate.format": "x509-pem-v1",
            "mega.trust.certificate.leaf_pem": leaf_pem,
            "mega.trust.signature.algorithm": "sha256-rsa-pkcs1",
            "mega.trust.signature.statement": statement.decode("utf-8"),
            "mega.trust.signature.value": signature_b64,
        },
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    with pytest.warns(RuntimeWarning, match="source risk: statement_invalid"):
        result = meta.verify_trust(root_pem)
    assert result["trusted"] is False
    assert result["risk"] == "statement_invalid"


def test_mega_unsigned_artifact_reports_source_risk(tmp_path, framework):
    path = os.path.join(tmp_path, "unsigned.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    with pytest.warns(RuntimeWarning, match="source risk: unsigned"):
        result = meta.verify_trust("", warn=True)
    assert result == {
        "trusted": False,
        "risk": "unsigned",
        "source_risk": True,
    }


def test_mega_non_raw_storage_format_is_not_mapped_as_dense(tmp_path, framework):
    path = os.path.join(tmp_path, "quantized.mega")
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 3],
                "logical_dtype": "F32",
                "storage_format": "blocked_quantized",
                "logical_nbytes": 24,
                "stored_nbytes": 3,
                "data": b"\x00\x01\x02",
            }
        ],
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    assert meta.tensors["layers.0.weight"].storage_format == "blocked_quantized"
    with pytest.raises(NotImplementedError, match="storage_format=blocked_quantized"):
        with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
            f.get_tensor("layers.0.weight")


def test_mega_zstd_byte_shuffled_raw_dense_loads_via_cpp(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "zstd-shuffled.mega")
    values = [1.25, 2.5, 3.75, 5.0, 6.25, 7.5]
    raw = struct.pack("<6f", *values)
    encoded = zstd.ZstdCompressor(level=3).compress(_byte_shuffle(raw, 4))
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 3],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": len(encoded),
                "tensor_flags": TENSOR_FLAG_COMPRESSED | TENSOR_FLAG_BYTE_SHUFFLED,
                "compression_codec": COMPRESSION_ZSTD,
                "shuffle_elem_size": 4,
                "data": encoded,
            }
        ],
    )

    expected = torch.tensor(values, dtype=torch.float32).reshape(2, 3)
    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        actual = f.get_tensor("layers.0.weight").clone()
    assert torch.equal(actual, expected)


def test_mega_whole_tensor_crc32_checksum_loads_via_cpp(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "tensor-crc32.mega")
    values = [2.0, 4.0, 6.0, 8.0]
    raw = struct.pack("<4f", *values)
    encoded = zstd.ZstdCompressor(level=3).compress(raw)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": len(encoded),
                "tensor_flags": TENSOR_FLAG_COMPRESSED,
                "compression_codec": COMPRESSION_ZSTD,
                "checksum_type": CHECKSUM_CRC32,
                "data": encoded,
            }
        ],
    )

    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        actual = f.get_tensor("layers.0.weight").clone()
    expected = torch.tensor(values, dtype=torch.float32).reshape(2, 2)
    assert torch.equal(actual, expected)


def test_mega_whole_tensor_crc32_checksum_rejects_mismatch(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-tensor-crc32.mega")
    raw = struct.pack("<2f", 1.0, 2.0)
    encoded = zstd.ZstdCompressor(level=3).compress(raw)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": len(encoded),
                "tensor_flags": TENSOR_FLAG_COMPRESSED,
                "compression_codec": COMPRESSION_ZSTD,
                "checksum_type": CHECKSUM_CRC32,
                "checksum": struct.pack("<I", 0xDEADBEEF) + b"\0" * 28,
                "data": encoded,
            }
        ],
    )

    with pytest.raises(Exception, match="checksum mismatch"):
        with mega_open(path, framework="pt", device="cpu", nogds=True):
            pass


def test_mega_whole_tensor_sha256_checksum_loads_via_cpp(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "tensor-sha256.mega")
    values = [1.0, 2.0, 4.0, 8.0]
    raw = struct.pack("<4f", *values)
    encoded = zstd.ZstdCompressor(level=3).compress(raw)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": len(encoded),
                "tensor_flags": TENSOR_FLAG_COMPRESSED,
                "compression_codec": COMPRESSION_ZSTD,
                "checksum_type": CHECKSUM_SHA256,
                "data": encoded,
            }
        ],
    )

    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        actual = f.get_tensor("layers.0.weight").clone()
    expected = torch.tensor(values, dtype=torch.float32).reshape(2, 2)
    assert torch.equal(actual, expected)


def test_mega_whole_tensor_sha256_checksum_rejects_mismatch(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-tensor-sha256.mega")
    raw = struct.pack("<2f", 1.0, 2.0)
    encoded = zstd.ZstdCompressor(level=3).compress(raw)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": len(encoded),
                "tensor_flags": TENSOR_FLAG_COMPRESSED,
                "compression_codec": COMPRESSION_ZSTD,
                "checksum_type": CHECKSUM_SHA256,
                "checksum": b"\xff" * 32,
                "data": encoded,
            }
        ],
    )

    with pytest.raises(Exception, match="checksum mismatch"):
        with mega_open(path, framework="pt", device="cpu", nogds=True):
            pass


def test_mega_chunked_raw_dense_loads_via_cpp(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "chunked.mega")
    values = [float(i) for i in range(8)]
    raw = struct.pack("<8f", *values)
    first = raw[:16]
    second = raw[16:]
    encoded_second = zstd.ZstdCompressor(level=3).compress(_byte_shuffle(second, 4))
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 4],
                "logical_dtype": "F32",
                "storage_format": "chunked_raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": 0,
                "shuffle_elem_size": 4,
                "chunks": [
                    {
                        "logical_offset": 0,
                        "logical_size": len(first),
                        "data": first,
                    },
                    {
                        "logical_offset": len(first),
                        "logical_size": len(second),
                        "data": encoded_second,
                        "flags": TENSOR_FLAG_COMPRESSED
                        | TENSOR_FLAG_BYTE_SHUFFLED,
                        "codec": COMPRESSION_ZSTD,
                    },
                ],
            }
        ],
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    frame = meta.tensors["layers.0.weight"]
    assert frame.storage_format == "chunked_raw_dense"
    assert len(frame.chunks) == 2

    expected = torch.tensor(values, dtype=torch.float32).reshape(2, 4)
    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        actual = f.get_tensor("layers.0.weight").clone()
    assert torch.equal(actual, expected)


def test_mega_chunked_raw_dense_rejects_incomplete_chunks(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-chunked.mega")
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "chunked_raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": 0,
                "chunks": [
                    {
                        "logical_offset": 0,
                        "logical_size": len(raw) // 2,
                        "data": raw[: len(raw) // 2],
                    }
                ],
            }
        ],
    )

    with pytest.raises(Exception, match="chunks do not cover logical tensor bytes"):
        MegaTensorsMetadata.from_file(path, framework)


def test_mega_chunk_crc32_checksum_loads_via_cpp(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "chunk-crc32.mega")
    values = [float(i) for i in range(4)]
    raw = struct.pack("<4f", *values)
    encoded = zstd.ZstdCompressor(level=3).compress(raw)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "chunked_raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": 0,
                "chunks": [
                    {
                        "logical_offset": 0,
                        "logical_size": len(raw),
                        "data": encoded,
                        "flags": TENSOR_FLAG_COMPRESSED,
                        "codec": COMPRESSION_ZSTD,
                        "checksum_type": CHECKSUM_CRC32,
                    }
                ],
            }
        ],
    )

    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        actual = f.get_tensor("layers.0.weight").clone()
    expected = torch.tensor(values, dtype=torch.float32).reshape(2, 2)
    assert torch.equal(actual, expected)


def test_mega_chunk_crc32_checksum_rejects_mismatch(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-crc32.mega")
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "chunked_raw_dense",
                "logical_nbytes": len(raw),
                "stored_nbytes": 0,
                "chunks": [
                    {
                        "logical_offset": 0,
                        "logical_size": len(raw),
                        "data": raw,
                        "checksum_type": CHECKSUM_CRC32,
                        "checksum": struct.pack("<I", 0xDEADBEEF) + b"\0" * 28,
                    }
                ],
            }
        ],
    )

    with pytest.raises(Exception, match="checksum mismatch"):
        with mega_open(path, framework="pt", device="cpu", nogds=True):
            pass


def test_mega_segment_directory_tracks_mixed_storage_without_changing_values(
    tmp_path, framework
):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "segmented.mega")
    raw_values = [1.0, 2.0, 3.0, 4.0]
    chunked_values = [5.0, 6.0, 7.0, 8.0]
    raw_payload = struct.pack("<4f", *raw_values)
    chunked_payload = struct.pack("<4f", *chunked_values)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.raw",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(raw_payload),
                "data": raw_payload,
            },
            {
                "name": "layers.0.chunked",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "chunked_raw_dense",
                "logical_nbytes": len(chunked_payload),
                "stored_nbytes": 0,
                "chunks": [
                    {
                        "logical_offset": 0,
                        "logical_size": len(chunked_payload),
                        "data": zstd.ZstdCompressor(level=3).compress(chunked_payload),
                        "flags": TENSOR_FLAG_COMPRESSED,
                        "codec": COMPRESSION_ZSTD,
                    }
                ],
            },
        ],
        segments=[
            {
                "segment_id": 7,
                "kind": 2,
                "priority": 10,
                "layer_start": 0,
                "layer_end": 1,
                "payload_offset": "auto",
                "payload_size": "auto",
                "first_tensor_id": 0,
                "tensor_count": 2,
                "prefetch_group": 1,
            }
        ],
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    assert meta.segments == [
        {
            "segment_id": 7,
            "kind": 2,
            "priority": 10,
            "flags": 0,
            "layer_start": 0,
            "layer_end": 1,
            "expert_start": 0,
            "expert_end": 0,
            "payload_offset": 0,
            "payload_size": meta.size_bytes - meta.header_length,
            "first_tensor_id": 0,
            "tensor_count": 2,
            "prefetch_group": 1,
            "device_hint": 0,
            "cache_policy": 0,
        }
    ]
    assert meta.select_segment_byte_ranges(lambda s: s["priority"] <= 10) == [
        (meta.header_length, meta.size_bytes)
    ]

    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        assert torch.equal(
            f.get_tensor("layers.0.raw").clone(),
            torch.tensor(raw_values, dtype=torch.float32).reshape(2, 2),
        )
        assert torch.equal(
            f.get_tensor("layers.0.chunked").clone(),
            torch.tensor(chunked_values, dtype=torch.float32).reshape(2, 2),
        )


def test_mega_segment_filter_stages_visible_tensors_without_decoding_cold_chunks(
    tmp_path, framework
):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "staged-segments.mega")
    warm_values = [1.0, 2.0]
    cold_values = [3.0, 4.0]
    warm_payload = struct.pack("<2f", *warm_values)
    cold_payload = zstd.ZstdCompressor(level=3).compress(struct.pack("<2f", *cold_values))
    _write_mega(
        path,
        [
            {
                "name": "layers.0.warm",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(warm_payload),
                "data": warm_payload,
            },
            {
                "name": "layers.9.cold",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "chunked_raw_dense",
                "logical_nbytes": 8,
                "stored_nbytes": 0,
                "chunks": [
                    {
                        "logical_offset": 0,
                        "logical_size": 8,
                        "data": cold_payload,
                        "flags": TENSOR_FLAG_COMPRESSED,
                        "codec": COMPRESSION_ZSTD,
                        "checksum_type": CHECKSUM_CRC32,
                        "checksum": struct.pack("<I", 0xDEADBEEF) + b"\0" * 28,
                    }
                ],
            },
        ],
        segments=[
            {
                "segment_id": 0,
                "kind": 2,
                "priority": 10,
                "payload_offset": 0,
                "payload_size": len(warm_payload),
                "first_tensor_id": 0,
                "tensor_count": 1,
            },
            {
                "segment_id": 1,
                "kind": 7,
                "priority": 50,
                "payload_offset": len(warm_payload),
                "payload_size": len(cold_payload),
                "first_tensor_id": 1,
                "tensor_count": 1,
            },
        ],
    )

    loader = MegaTensorsFileLoader(None, "cpu", nogds=True, framework="pytorch")
    loader.set_segment_filter(lambda segment: segment["priority"] <= 10)
    loader.add_filenames({0: [path]})
    assert loader.get_keys() == ["layers.0.warm"]
    fb = loader.copy_files_to_device()
    try:
        assert list(fb.key_to_rank_lidx.keys()) == ["layers.0.warm"]
        assert torch.equal(
            fb.get_tensor("layers.0.warm").clone(),
            torch.tensor(warm_values, dtype=torch.float32),
        )
        with pytest.raises(ValueError, match="was not found"):
            fb.get_tensor("layers.9.cold")
    finally:
        fb.close()
        loader.close()

    with pytest.raises(Exception, match="checksum mismatch"):
        with mega_open(path, framework="pt", device="cpu", nogds=True):
            pass


def test_mega_segment_directory_rejects_invalid_tensor_range(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-segment.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.weight",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        segments=[
            {
                "segment_id": 0,
                "payload_offset": "auto",
                "payload_size": "auto",
                "first_tensor_id": 1,
                "tensor_count": 1,
            }
        ],
    )

    with pytest.raises(Exception, match="segment tensor range exceeds tensor directory"):
        MegaTensorsMetadata.from_file(path, framework)


def test_mega_moe_expert_table_is_advisory_over_raw_tensors(tmp_path, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("test uses torch tensor assertions")
    import torch

    path = os.path.join(tmp_path, "moe.mega")
    gate = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    up = struct.pack("<4f", 5.0, 6.0, 7.0, 8.0)
    _write_mega(
        path,
        [
            {
                "name": "layers.3.experts.1.gate",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(gate),
                "data": gate,
            },
            {
                "name": "layers.3.experts.1.up",
                "shape": [2, 2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(up),
                "data": up,
            },
        ],
        metadata={
            "mega.moe.n_experts": 4,
            "mega.moe.n_experts_used": 2,
        },
        segments=[
            {
                "segment_id": 11,
                "kind": 4,
                "priority": 40,
                "layer_start": 3,
                "layer_end": 3,
                "expert_start": 1,
                "expert_end": 1,
                "payload_offset": "auto",
                "payload_size": "auto",
                "first_tensor_id": 0,
                "tensor_count": 2,
                "prefetch_group": 5,
                "cache_policy": 2,
            }
        ],
        moe_experts=[
            {
                "tensor_id": 0,
                "layer_id": 3,
                "expert_id": 1,
                "expert_role": MOE_ROLE_GATE,
                "segment_id": 11,
                "prefetch_group": 5,
                "cache_policy": 2,
            },
            {
                "tensor_id": 1,
                "layer_id": 3,
                "expert_id": 1,
                "expert_role": MOE_ROLE_UP,
                "segment_id": 11,
                "prefetch_group": 5,
                "cache_policy": 2,
            },
        ],
    )

    meta = MegaTensorsMetadata.from_file(path, framework)
    assert meta.get_moe_expert_tensors(layer_id=3, expert_id=1) == [
        "layers.3.experts.1.gate",
        "layers.3.experts.1.up",
    ]
    assert meta.moe_experts[0]["expert_role"] == MOE_ROLE_GATE

    with mega_open(path, framework="pt", device="cpu", nogds=True) as f:
        assert torch.equal(
            f.get_tensor("layers.3.experts.1.gate").clone(),
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).reshape(2, 2),
        )
        assert torch.equal(
            f.get_tensor("layers.3.experts.1.up").clone(),
            torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32).reshape(2, 2),
        )


def test_mega_moe_expert_table_rejects_unknown_segment(tmp_path, framework):
    path = os.path.join(tmp_path, "bad-moe.mega")
    payload = struct.pack("<2f", 1.0, 2.0)
    _write_mega(
        path,
        [
            {
                "name": "layers.0.experts.0.gate",
                "shape": [2],
                "logical_dtype": "F32",
                "storage_format": "raw_dense",
                "logical_nbytes": len(payload),
                "data": payload,
            }
        ],
        metadata={"mega.moe.n_experts": 1},
        segments=[
            {
                "segment_id": 1,
                "payload_offset": "auto",
                "payload_size": "auto",
                "first_tensor_id": 0,
                "tensor_count": 1,
            }
        ],
        moe_experts=[
            {
                "tensor_id": 0,
                "layer_id": 0,
                "expert_id": 0,
                "expert_role": MOE_ROLE_GATE,
                "segment_id": 2,
            }
        ],
    )

    with pytest.raises(Exception, match="unknown segment_id"):
        MegaTensorsMetadata.from_file(path, framework)
