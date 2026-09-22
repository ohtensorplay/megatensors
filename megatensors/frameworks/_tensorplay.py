# SPDX-License-Identifier: Apache-2.0

"""TensorPlay framework support for the MEGA tensor loader.

TensorPlay is intentionally imported lazily by :func:`get_framework_op` so
installing megatensors does not make TensorPlay a hard dependency.  The
adapter keeps the MEGA loader's framework-neutral path intact and only maps
devices, dtypes, DLPack, and tensor allocation to TensorPlay.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

try:
    import tensorplay as tp
except ImportError as exc:  # pragma: no cover - exercised by import diagnostics
    raise ImportError(
        "TensorPlay support requires the 'tensorplay' package. "
        "Install TensorPlay or select framework='pt'/'paddle'."
    ) from exc

from .. import cpp as megacpp
from ..common import SingleGroup
from ..cpp import cpu_free, cpu_malloc, gds_device_buffer
from ..st_types import Device, DeviceType, DType
from . import FrameworkOpBase, ProcessGroupBase, TensorBase


_TP_UINT32 = getattr(tp, "uint32", tp._C.DType.uint32)
_TP_UINT64 = getattr(tp, "uint64", tp._C.DType.uint64)


_TP_DTYPE = {
    DType.BOOL: tp.bool,
    DType.U8: tp.uint8,
    DType.I8: tp.int8,
    DType.I16: tp.int16,
    DType.I32: tp.int32,
    DType.I64: tp.int64,
    DType.U16: tp.uint16,
    DType.U32: _TP_UINT32,
    DType.U64: _TP_UINT64,
    DType.F16: tp.float16,
    DType.BF16: tp.bfloat16,
    DType.F32: tp.float32,
    DType.F64: tp.float64,
}

_TP_DTYPE_NAME = {
    tp.bool: "BOOL",
    tp.uint8: "U8",
    tp.int8: "I8",
    tp.int16: "I16",
    tp.int32: "I32",
    tp.int64: "I64",
    tp.uint16: "U16",
    _TP_UINT32: "U32",
    _TP_UINT64: "U64",
    tp.float16: "F16",
    tp.bfloat16: "BF16",
    tp.float32: "F32",
    tp.float64: "F64",
}

_TP_DTYPE_SIZE = {
    DType.BOOL: 1,
    DType.U8: 1,
    DType.I8: 1,
    DType.I16: 2,
    DType.I32: 4,
    DType.I64: 8,
    DType.U16: 2,
    DType.U32: 4,
    DType.U64: 8,
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.F64: 8,
}


def _tp_dtype(dtype: DType):
    try:
        return _TP_DTYPE[dtype]
    except KeyError as exc:
        raise NotImplementedError(
            f"TensorPlay does not expose MEGA dtype {dtype.value}"
        ) from exc


def _tp_device(device: Device):
    if device.type == DeviceType.CPU:
        return tp.Device(tp.DeviceType.CPU)
    if device.type == DeviceType.CUDA:
        return tp.Device(tp.DeviceType.CUDA, device.index or 0)
    raise NotImplementedError(f"TensorPlay does not support device {device.type.value}")


def _device_from_tensor(tensor: Any) -> Device:
    raw_device = tensor.device
    if raw_device.type == tp.DeviceType.CPU:
        return Device(DeviceType.CPU)
    return Device(DeviceType.CUDA, raw_device.index)


@dataclass
class TensorPlayTensor(TensorBase):
    """A MEGA framework wrapper around a native ``tensorplay.Tensor``."""

    real_tensor: Any

    def get_raw(self) -> Any:
        return self.real_tensor

    def contiguous(self) -> "TensorPlayTensor":
        if self.real_tensor.is_contiguous():
            return self
        return TensorPlayTensor(self.device, self.dtype, self.real_tensor.clone())

    def to(
        self,
        device: Optional[Device] = None,
        dtype: DType = DType.AUTO,
    ) -> "TensorPlayTensor":
        target_device = device or self.device
        target_dtype = dtype if dtype != DType.AUTO else self.dtype
        if device is None and dtype == DType.AUTO:
            return self

        raw = self.real_tensor
        if device is not None and device != self.device:
            raw = raw.to(_tp_device(device))
        if dtype != DType.AUTO and dtype != self.dtype:
            raw = raw.to(_tp_dtype(dtype))
        return TensorPlayTensor(target_device, target_dtype, raw)

    def clone(self) -> "TensorPlayTensor":
        return TensorPlayTensor(self.device, self.dtype, self.real_tensor.clone())

    def detach(self) -> "TensorPlayTensor":
        return TensorPlayTensor(self.device, self.dtype, self.real_tensor.detach())

    def view(self, dtype: DType) -> "TensorPlayTensor":
        return TensorPlayTensor(self.device, dtype, self.real_tensor.view(_tp_dtype(dtype)))

    def __getitem__(self, value: Any) -> "TensorPlayTensor":
        return TensorPlayTensor(self.device, self.dtype, self.real_tensor[value])

    def reshape(self, shape: List[int]) -> "TensorPlayTensor":
        return TensorPlayTensor(self.device, self.dtype, self.real_tensor.reshape(shape))

    def data_ptr(self) -> int:
        return int(self.real_tensor.data_ptr())


@dataclass
class TensorPlayProcessGroup(ProcessGroupBase[TensorPlayTensor]):
    """Single-process group used by the TensorPlay adapter."""

    real_pg: Optional[Any] = None

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def _unsupported(self) -> None:
        if self.real_pg is not None:
            raise NotImplementedError("TensorPlay distributed MEGA loading is not implemented")

    def broadcast(self, dst: TensorPlayTensor, rank: int) -> None:
        self._unsupported()

    def scatter(
        self,
        dst: TensorPlayTensor,
        scatter_list: List[TensorPlayTensor],
        src: int,
    ) -> None:
        self._unsupported()

    def send(self, tensor: TensorPlayTensor, dst_rank: int, tag: int) -> None:
        self._unsupported()

    def recv(self, tensor: TensorPlayTensor, src_rank: int, tag: int) -> None:
        self._unsupported()


class TensorPlayOp(FrameworkOpBase[TensorPlayTensor, TensorPlayProcessGroup]):
    def __init__(self) -> None:
        self.mem_used = 0
        self._cuda_buffers: dict[int, Any] = {}

    def get_name(self) -> str:
        return "tensorplay"

    def get_device(self, device: str, pg: TensorPlayProcessGroup) -> Device:
        parsed = Device.from_str(device)
        if parsed.type == DeviceType.GPU:
            parsed = Device(DeviceType.CUDA, parsed.index)
        if parsed.type == DeviceType.CUDA and not tp.cuda.is_available():
            raise RuntimeError("TensorPlay CUDA support is not available")
        return parsed

    def set_device(self, device: Device) -> None:
        if device.type == DeviceType.CUDA:
            tp.cuda.set_device(device.index or 0)

    def alloc_tensor_memory(self, length: int, dev: Device) -> gds_device_buffer:
        if dev.type == DeviceType.CPU:
            ptr = cpu_malloc(length)
            self.mem_used += length
            return gds_device_buffer(ptr, length, False)
        if dev.type == DeviceType.CUDA:
            owner = tp.empty(
                (length,),
                dtype=tp.uint8,
                device=_tp_device(dev),
            )
            ptr = int(owner.data_ptr())
            self._cuda_buffers[ptr] = owner
            self.mem_used += length
            return gds_device_buffer(ptr, length, True)
        raise NotImplementedError(f"TensorPlay does not support device {dev.type.value}")

    def free_tensor_memory(self, gbuf: gds_device_buffer, dev: Device) -> None:
        length = gbuf.get_length()
        ptr = int(gbuf.get_base_address())
        if dev.type == DeviceType.CPU:
            cpu_free(ptr)
        elif dev.type == DeviceType.CUDA:
            self._cuda_buffers.pop(ptr, None)
        else:
            raise NotImplementedError(f"TensorPlay does not support device {dev.type.value}")
        self.mem_used -= length

    def get_empty_tensor(
        self, shape: List[int], dtype: DType, device: Device
    ) -> TensorPlayTensor:
        return TensorPlayTensor(
            device,
            dtype,
            tp.empty(shape, dtype=_tp_dtype(dtype), device=_tp_device(device)),
        )

    def concat_tensors(
        self, tensors: List[TensorPlayTensor], dim: int
    ) -> TensorPlayTensor:
        # TensorPlay exposes cat as a top-level operation.
        raw = tp.cat([tensor.real_tensor for tensor in tensors], dim=dim)
        return TensorPlayTensor(tensors[0].device, tensors[0].dtype, raw)

    def get_dtype_size(self, dtype: DType) -> float:
        try:
            return float(_TP_DTYPE_SIZE[dtype])
        except KeyError as exc:
            raise NotImplementedError(
                f"TensorPlay does not expose MEGA dtype {dtype.value}"
            ) from exc

    def from_dlpack(
        self, dl_tensor: Any, device: Device, dtype: DType
    ) -> TensorPlayTensor:
        return TensorPlayTensor(device, dtype, tp.from_dlpack(dl_tensor))

    def copy_tensor(self, dst: TensorPlayTensor, src: TensorPlayTensor) -> None:
        dst.real_tensor.copy_(src.real_tensor)

    def get_cuda_ver(self) -> str:
        # TensorPlay owns its CUDA runtime.  Do not make megatensors load a
        # second runtime merely because TensorPlay was built with CUDA; the
        # MEGA CPU path must also remain usable on a host where megatensors'
        # optional CUDA runtime is unavailable.
        return "cuda-tensorplay" if megacpp.is_cuda_found() else "0.0"

    def get_device_ptr_align(self) -> int:
        return 16

    def as_workaround_dtype(self, dtype: DType) -> DType:
        # TensorPlay has native representations for the dtypes it advertises.
        # Do not silently reinterpret unsupported FP8/FP4 weights.
        _tp_dtype(dtype)
        return dtype

    def synchronize(self, device: Device) -> None:
        if device.type == DeviceType.CUDA:
            tp.cuda.synchronize()

    def get_process_group(self, pg: Optional[Any]) -> TensorPlayProcessGroup:
        if pg is None or isinstance(pg, SingleGroup):
            return TensorPlayProcessGroup()
        raise NotImplementedError("TensorPlay distributed MEGA loading is not implemented")

    def is_equal(self, wrapped: TensorPlayTensor, real: Any) -> bool:
        if isinstance(real, tp.Tensor):
            return bool(tp.allclose(wrapped.real_tensor, real))
        raise TypeError("real is not a tensorplay.Tensor")

    def randn(self, s: tuple, device: Device, dtype: DType) -> TensorPlayTensor:
        return TensorPlayTensor(
            device,
            dtype,
            tp.randn(s, dtype=_tp_dtype(dtype), device=_tp_device(device)),
        )

    def support_fp8(self) -> bool:
        return False

    def get_mem_used(self) -> int:
        return self.mem_used

    def get_device_name(self, index: int) -> str:
        return ""

    def mmap_file_pinned(
        self, filename: str, length: int, offset: int
    ) -> TensorPlayTensor:
        # TensorPlay currently has no pinned-host allocator.  A normal CPU
        # tensor is still correct and keeps the unified copier functional.
        raw = tp.Tensor._load_file_segment(
            filename,
            offset,
            length,
            [length],
            tp.uint8,
        )
        return TensorPlayTensor(Device(DeviceType.CPU), DType.U8, raw)


_op: Optional[TensorPlayOp] = None


def get_framework_op() -> FrameworkOpBase:
    global _op
    if _op is None:
        _op = TensorPlayOp()
    return _op


def _normalize_tensor_items(
    tensors: Any,
) -> tuple[list[tuple[str, Any]], str]:
    if isinstance(tensors, tp.Tensor):
        return [("0", tensors)], "tensor"
    if isinstance(tensors, Mapping):
        return [(str(name), value) for name, value in tensors.items()], "dict"
    if isinstance(tensors, tuple):
        return [(str(index), value) for index, value in enumerate(tensors)], "tuple"
    if isinstance(tensors, list):
        return [(str(index), value) for index, value in enumerate(tensors)], "list"
    raise TypeError(
        "TensorPlay MEGA serialization accepts a tensor, mapping, tuple, or list"
    )


def _device_string_of(tensor: Any) -> str:
    raw_device = tensor.device
    if raw_device.type == tp.DeviceType.CPU:
        return "cpu"
    if raw_device.type == tp.DeviceType.CUDA:
        return f"cuda:{raw_device.index or 0}"
    return str(raw_device)


def _hash_payload_regions(
    payload_path: Path,
    regions: List[tuple[int, int]],
    checksum: str,
) -> List[bytes]:
    """Return raw digest bytes for ``(offset, nbytes)`` regions of ``payload_path``.

    CRC32 digests are packed little-endian (``<I``) to match the MEGA loader's
    ``read_checksum_u32_le``; SHA256 digests are the raw 32-byte hash.
    """

    import hashlib
    import struct
    import zlib

    digests: List[bytes] = []
    with open(payload_path, "rb") as handle:
        for offset, nbytes in regions:
            handle.seek(offset)
            if checksum == "crc32":
                crc = 0
                remaining = nbytes
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        raise IOError(f"short read while hashing {payload_path}")
                    crc = zlib.crc32(chunk, crc)
                    remaining -= len(chunk)
                digests.append(struct.pack("<I", crc & 0xFFFFFFFF))
            elif checksum == "sha256":
                hasher = hashlib.sha256()
                remaining = nbytes
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        raise IOError(f"short read while hashing {payload_path}")
                    hasher.update(chunk)
                    remaining -= len(chunk)
                digests.append(hasher.digest())
            else:
                raise ValueError(f"unsupported MEGA checksum mode: {checksum!r}")
    return digests


def write_tensorplay_file(
    filename: str,
    tensors: Mapping[str, Any] | Sequence[Any],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    alignment: int = 4096,
    checksum: str = "crc32",
    container: Optional[str] = None,
    layout: Optional[Any] = None,
) -> None:
    """Write TensorPlay tensors as a self-describing MEGA ``.mega`` file.

    The TensorPlay C++ binding exports raw tensor bytes while the MEGA C++
    extension writes the canonical header and payload layout.  This preserves
    BF16 and non-NumPy dtypes bit-for-bit instead of routing through
    ``Tensor.numpy()``.

    Tensors backed by the same storage (identical data pointer and extent,
    with a contiguous representative) are written once and referenced by every
    descriptor, so tied weights do not duplicate payload bytes.  Each unique
    payload region carries a CRC32 (or SHA256, or none) checksum that the MEGA
    loader verifies on load.

    ``container`` overrides the recorded ``tensorplay.container`` kind and
    ``layout`` (a JSON-serializable tree with ``{"__tensor__": name}`` leaves)
    records how to reassemble nested containers on load.
    """

    if checksum not in {"none", "crc32", "sha256"}:
        raise ValueError(
            f"MEGA checksum must be 'none', 'crc32', or 'sha256' (got {checksum!r})"
        )

    items, derived_container = _normalize_tensor_items(tensors)
    container_kind = container or derived_container
    for name, value in items:
        if not isinstance(value, tp.Tensor):
            raise TypeError(f"TensorPlay MEGA value {name!r} is not a TensorPlay tensor")
        if value.dtype not in _TP_DTYPE_NAME:
            raise NotImplementedError(f"TensorPlay dtype {value.dtype} is not MEGA-compatible")

    # Group items by backing storage so shared payloads are written once.
    # Only exact matches (data_ptr, nbytes) with a contiguous representative
    # are safe; anything else gets its own dense region.
    group_rep: List[Any] = []
    group_offset: List[int] = []
    item_region_size: List[int] = []
    item_group: List[int] = []
    group_keys: dict[tuple[int, int], int] = {}
    for _, value in items:
        nbytes = int(value.numel()) * int(value.itemsize())
        item_region_size.append(nbytes)
        key: Optional[tuple[int, int]] = None
        if nbytes > 0 and value.is_contiguous():
            key = (int(value.data_ptr()), nbytes)
            if key in group_keys:
                item_group.append(group_keys[key])
                continue
        if key is None:
            group_id = len(group_rep)
        else:
            group_id = len(group_rep)
            group_keys[key] = group_id
        item_group.append(group_id)
        group_rep.append(value)

    file_metadata = dict(metadata or {})
    devices = {_device_string(value) for _, value in items}
    file_metadata.update(
        {
            "general.architecture": "tensorplay",
            "general.alignment": int(alignment),
            "mega.layout.tensor_order": "original",
            "mega.tensor_info.format": "self_describing",
            "tensorplay.container": container_kind,
            "tensorplay.format.version": 1,
            "tensorplay.byteorder": sys.byteorder,
        }
    )
    if layout is not None:
        file_metadata["tensorplay.layout"] = json.dumps(layout, separators=(",", ":"))
    if any(device != "cpu" for device in devices):
        file_metadata["tensorplay.tensor_devices"] = json.dumps(
            {name: _device_string(value) for name, value in items},
            separators=(",", ":"),
        )
    if checksum != "none":
        file_metadata["tensorplay.checksum"] = checksum

    with tempfile.TemporaryDirectory(prefix="tensorplay-mega-") as temp_dir:
        payload_path = Path(temp_dir) / "payload.bin"
        payload_offset = 0
        for rep in group_rep:
            tp.Tensor._save_file_segments(str(payload_path), [rep])
            group_offset.append(payload_offset)
            payload_offset += int(rep.numel()) * int(rep.itemsize())

        descriptors = []
        if checksum != "none":
            digests = _hash_payload_regions(
                payload_path,
                [
                    (group_offset[gid], item_region_size[index])
                    for index, gid in enumerate(item_group)
                ],
                checksum,
            )
        else:
            digests = [b""] * len(items)
        for index, (name, value) in enumerate(items):
            gid = item_group[index]
            nbytes = item_region_size[index]
            descriptor = {
                "name": name,
                "shape": [int(dim) for dim in value.shape],
                "logical_dtype": _TP_DTYPE_NAME[value.dtype],
                "storage_format": "raw_dense",
                "payload_offset": group_offset[gid],
                "logical_nbytes": nbytes,
                "stored_nbytes": nbytes,
                "src_filename": str(payload_path),
                "src_offset": group_offset[gid],
            }
            if checksum == "crc32":
                descriptor["checksum_type"] = 1
                descriptor["checksum"] = digests[index].ljust(32, b"\0")
            elif checksum == "sha256":
                descriptor["checksum_type"] = 2
                descriptor["checksum"] = digests[index]
            descriptors.append(descriptor)
        megacpp.write_file(str(filename), descriptors, file_metadata, int(alignment))
