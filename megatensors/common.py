# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from math import prod
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from . import cpp as megacpp
from .dlpack import from_cuda_buffer
from .frameworks import FrameworkOpBase, TensorBase
from .st_types import Device, DeviceType, DType


MEGA_TENSOR_INFO_FORMAT = "self_describing"
MEGA_RAW_DENSE_STORAGE_FORMAT = "raw_dense"
MEGA_CHUNKED_RAW_DENSE_STORAGE_FORMAT = "chunked_raw_dense"
MEGA_TENSOR_FLAG_COMPRESSED = 1
MEGA_TENSOR_FLAG_BYTE_SHUFFLED = 2
MEGA_KNOWN_TENSOR_FLAGS = MEGA_TENSOR_FLAG_COMPRESSED | MEGA_TENSOR_FLAG_BYTE_SHUFFLED
MEGA_COMPRESSION_NONE = 0
MEGA_COMPRESSION_ZSTD = 1
MEGA_TRUST_CERTIFICATE_FORMAT = "x509-pem-v1"
MEGA_TRUST_SIGNATURE_STATEMENT_PREFIX = "MEGA-TRUST-v1\n"
MEGA_TRUST_STATEMENT_KEYS = (
    "format_version",
    "artifact_kind",
    "publisher",
    "model_id",
    "payload_sha256",
    "created_at",
    "expires_at",
)
MEGAKV_KEY_ROLE = 1
MEGAKV_VALUE_ROLE = 2


@dataclass(frozen=True)
class MegaModelInfo:
    architecture: str
    name: str = ""
    basename: str = ""
    size_label: str = ""
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    feed_forward_length: int = 0
    attention_head_count: int = 0
    attention_head_count_kv: int = 0
    rope_dimension_count: int = 0
    vocab_size: int = 0


@dataclass(frozen=True)
class MegaTokenizerInfo:
    model: str
    tokens: List[str]
    token_types: List[int]
    merges: List[str]
    bos_token_id: int = -1
    eos_token_id: int = -1
    unknown_token_id: int = -1
    padding_token_id: int = -1
    bos_token: str = ""
    eos_token: str = ""
    unknown_token: str = ""
    padding_token: str = ""
    pre_tokenizer: str = ""
    add_bos_token: bool = False
    add_eos_token: bool = False


@dataclass(frozen=True)
class MegaKvEntry:
    entry_id: int
    name: str
    layer_id: int
    kv_role: int
    sequence_id: int
    token_start: int
    token_count: int
    shape: List[int]
    dtype: DType
    payload_offset: int
    logical_nbytes: int
    stored_nbytes: int
    flags: int
    codec: int
    shuffle_elem_size: int
    checksum_type: int
    checksum: bytes


def init_logger(name: str):
    return logging.getLogger(name)


def set_debug():
    logging.basicConfig(
        format="[%(levelname)s] %(message)s", level=logging.DEBUG, force=True
    )


def is_debug(logger: logging.Logger) -> bool:
    return logger.isEnabledFor(logging.DEBUG)


def is_gpu_found():
    """Check if any GPU (CUDA or HIP) is available.

    Returns True if either CUDA or ROCm/HIP GPUs are detected.
    This allows code to work transparently across both platforms.
    """
    return megacpp.is_cuda_found() or megacpp.is_hip_found()


def get_device_numa_node(device: Optional[int]) -> Optional[int]:
    if device is None or not sys.platform.startswith("linux"):
        return None
    pci_addr = megacpp.get_device_pci_bus(device)
    if pci_addr == "":
        return None
    bus_addr = ":".join(pci_addr.split(":")[:2]).lower()
    syspath = f"/sys/class/pci_bus/{bus_addr}/device/numa_node"
    if not os.path.exists(syspath):
        return None
    with open(syspath) as f:
        return int(f.read().strip())


def _normalize_windows_dll_path(path: str, source: str) -> str:
    """Return a normalized absolute DLL path on Windows.

    We intentionally reject relative paths and bare DLL names here so callers do
    not fall back to the Windows DLL search order, which is susceptible to DLL
    planting / search-order hijacking.
    """
    expanded = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(expanded):
        raise ValueError(f"{source} must be an absolute path on Windows: {path!r}")
    normalized = os.path.abspath(expanded)
    if not os.path.isfile(normalized):
        raise FileNotFoundError(
            f"{source} points to a missing DLL on Windows: {normalized}"
        )
    return normalized


def resolve_runtime_lib_name(framework=None) -> str:
    """Resolve the GPU runtime library to dlopen for the current platform.

    On Windows, returns an absolute cudart DLL path. On other platforms, maps the framework's declared GPU
    vendor to a runtime library so the dlopen'd vendor stays in sync with the
    framework's GPU build. Returns "" when there is no usable hint so the caller falls
    back to auto-detection.
    """
    if sys.platform == "win32":
        return _resolve_windows_cudart_lib_name()
    if framework is None:
        return ""
    try:
        ver = framework.get_cuda_ver()
    except Exception:
        return ""
    if not ver or "-" not in ver:
        return ""
    vendor = ver.split("-", 1)[0]
    if vendor == "hip":
        return "libamdhip64.so"
    if vendor == "cuda":
        return "libcudart.so"
    return ""


def _resolve_windows_cudart_lib_name() -> str:
    """Resolve the absolute cudart DLL path on Windows, "" for the default."""
    # Allow explicit override via environment variable
    override = os.environ.get("MEGATENSORS_CUDART_LIB", "").strip()
    if override:
        return _normalize_windows_dll_path(override, "MEGATENSORS_CUDART_LIB")

    import glob

    def _find_cudart_in_dir(d: str) -> str:
        """Scan a trusted directory for cudart64_*.dll files."""
        if not d:
            return ""
        expanded = os.path.expandvars(os.path.expanduser(d))
        if not os.path.isabs(expanded):
            return ""
        d = os.path.abspath(expanded)
        if not os.path.isdir(d):
            return ""
        matches = glob.glob(os.path.join(d, "cudart64_*.dll"))
        if matches:
            matches.sort(reverse=True)
            return os.path.abspath(matches[0])
        return ""

    def _detect_from_nvcc(cuda_home: str) -> str:
        """Try to detect the CUDA major version from nvcc -V output."""
        expanded = os.path.expandvars(os.path.expanduser(cuda_home))
        if not os.path.isabs(expanded):
            return ""
        cuda_home = os.path.abspath(expanded)
        nvcc = os.path.join(cuda_home, "bin", "nvcc.exe")
        if not os.path.isfile(nvcc):
            return ""
        try:
            import subprocess

            output = subprocess.check_output(
                [nvcc, "-V"], universal_newlines=True, stderr=subprocess.STDOUT
            )
            tokens = output.split()
            release_idx = tokens.index("release") + 1
            version_str = tokens[release_idx].rstrip(",")
            cuda_major = version_str.split(".")[0]
            candidate = os.path.join(cuda_home, "bin", f"cudart64_{cuda_major}.dll")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            return ""
        except Exception:
            return ""

    # Try to detect from CUDA_HOME / CUDA_PATH
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        result = _detect_from_nvcc(cuda_home)
        if result:
            return result
        result = _find_cudart_in_dir(os.path.join(cuda_home, "bin"))
        if result:
            return result

    # Scan common NVIDIA install locations
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    nvidia_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
    if os.path.isdir(nvidia_base):
        # List version directories (e.g. v12.6, v11.8), newest first
        try:
            versions = sorted(os.listdir(nvidia_base), reverse=True)
        except OSError:
            versions = []
        for ver_dir in versions:
            result = _find_cudart_in_dir(os.path.join(nvidia_base, ver_dir, "bin"))
            if result:
                return result

    return ""  # fall back to compiled-in default


# keep this for compatibility
class SingleGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class MegaTensorsMetadata:
    def __init__(
        self,
        parsed: Dict,
        size_bytes: int,
        framework: FrameworkOpBase,
        src: str = "",
        keep_orig_dict: bool = False,
    ):
        self.src = src
        self.framework = framework
        self.version = int(parsed["version"])
        self.metadata = dict(parsed["metadata"])
        tensor_records = parsed["tensor_records"]
        self.segments = list(parsed.get("segments", []))
        self.moe_experts = list(parsed.get("moe_experts", []))
        self.footer_overlay = dict(parsed.get("footer_overlay", {"present": False}))
        self.tensors: Dict[str, TensorFrame] = {}
        self.has_encoded_tensors = False
        self.header_length = int(parsed["header_length"])
        self.file_size_bytes = size_bytes
        self.payload_end = int(parsed.get("payload_end", size_bytes))
        if self.payload_end > size_bytes:
            raise Exception(
                f"{src}: payload_end exceeds file size, payload_end={self.payload_end}, size_bytes={size_bytes}"
            )
        self.aligned = self.header_length % framework.get_device_ptr_align() == 0
        if keep_orig_dict:
            self.ser = tensor_records

        if self.metadata.get("mega.tensor_info.format") != MEGA_TENSOR_INFO_FORMAT:
            raise Exception(
                f"{src}: missing or invalid mega.tensor_info.format={MEGA_TENSOR_INFO_FORMAT}"
            )

        for record in tensor_records:
            k, t = TensorFrame.from_record(record)
            self.tensors[k] = t
            if (
                t.storage_format != MEGA_RAW_DENSE_STORAGE_FORMAT
                or t.tensor_flags != 0
                or t.compression_codec != MEGA_COMPRESSION_NONE
            ):
                self.has_encoded_tensors = True
            s, e = t.stored_offsets
            if e < s:
                raise Exception(
                    f"validate(tensor {k}): InvalidOffset s={s}, e={e}, src={src}"
                )
            nelements = prod(t.shape) if len(t.shape) > 0 else 1
            nbytes = int(nelements * framework.get_dtype_size(t.dtype))
            if t.logical_nbytes != nbytes:
                raise Exception(
                    f"validate(tensor {k}): TensorInvalidInfo, logical_nbytes={t.logical_nbytes}, nbytes={nbytes}, src={src}"
                )
        self.size_bytes = self.payload_end
        max_stored_end = max(
            (frame.stored_offsets[1] for frame in self.tensors.values()), default=0
        )
        if max_stored_end + self.header_length > self.size_bytes:
            raise Exception(
                f"MetadataIncompleteBuffer, src={src}, payload_end={max_stored_end}, header_length={self.header_length}, size_bytes={self.size_bytes}"
            )
        if max_stored_end + self.header_length < self.size_bytes:
            trailing = self.size_bytes - (max_stored_end + self.header_length)
            logger = init_logger(__name__)
            logger.debug(
                "trailing %d bytes after tensor data in %s (alignment padding)",
                trailing,
                src,
            )

    def _load_encoded_raw_dense_tensor(
        self,
        tensor_name: str,
        t: "TensorFrame",
        device: Device,
        dtype: DType,
    ) -> TensorBase:
        if t.storage_format == MEGA_CHUNKED_RAW_DENSE_STORAGE_FORMAT:
            if not t.chunks and t.logical_nbytes > 0:
                raise Exception(f"{self.src}: tensor {tensor_name} has no chunks")
            t2 = self.framework.get_empty_tensor(t.shape, t.dtype, device)
            flags = os.O_RDONLY
            if sys.platform == "win32" and hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(self.src, flags, 0o644)
            try:
                megacpp.decode_chunks_fd(
                    fd,
                    self.src,
                    tensor_name,
                    self.header_length,
                    t.chunks,
                    t.logical_nbytes,
                    t.shuffle_elem_size,
                    t2.data_ptr(),
                    device.type != DeviceType.CPU,
                )
            finally:
                os.close(fd)
            if dtype != DType.AUTO and dtype != t.dtype:
                if self.framework.get_dtype_size(dtype) > self.framework.get_dtype_size(
                    t.dtype
                ):
                    raise Exception(
                        f"Online type conversion to larger sizes is not supported ({t.dtype} -> {dtype})"
                    )
                t2 = t2.to(dtype=dtype)
                self.tensors[tensor_name].dtype = dtype
            return t2

        if t.tensor_flags & ~MEGA_KNOWN_TENSOR_FLAGS:
            raise Exception(
                f"{self.src}: tensor {tensor_name} has unknown tensor_flags={t.tensor_flags}"
            )
        compressed = (t.tensor_flags & MEGA_TENSOR_FLAG_COMPRESSED) != 0
        byte_shuffled = (t.tensor_flags & MEGA_TENSOR_FLAG_BYTE_SHUFFLED) != 0
        if not compressed and t.compression_codec != MEGA_COMPRESSION_NONE:
            raise Exception(
                f"{self.src}: tensor {tensor_name} is not compressed but "
                f"compression_codec={t.compression_codec}"
            )
        if byte_shuffled and t.logical_nbytes % t.shuffle_elem_size != 0:
            raise Exception(
                f"{self.src}: tensor {tensor_name} logical_nbytes={t.logical_nbytes} "
                f"is not divisible by shuffle_elem_size={t.shuffle_elem_size}"
            )

        t2 = self.framework.get_empty_tensor(t.shape, t.dtype, device)
        flags = os.O_RDONLY
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(self.src, flags, 0o644)
        try:
            megacpp.decode_payload_fd(
                fd,
                self.src,
                tensor_name,
                self.header_length + t.stored_offsets[0],
                t.stored_nbytes,
                t.logical_nbytes,
                t.tensor_flags,
                t.compression_codec,
                t.shuffle_elem_size,
                t.checksum_type,
                t.checksum,
                t2.data_ptr(),
                device.type != DeviceType.CPU,
            )
        finally:
            os.close(fd)
        if dtype != DType.AUTO and dtype != t.dtype:
            if self.framework.get_dtype_size(dtype) > self.framework.get_dtype_size(
                t.dtype
            ):
                raise Exception(
                    f"Online type conversion to larger sizes is not supported ({t.dtype} -> {dtype})"
                )
            t2 = t2.to(dtype=dtype)
            self.tensors[tensor_name].dtype = dtype
        return t2

    @classmethod
    def from_buffer(
        self, buf: int, buffer_len: int, filename: str, framework: FrameworkOpBase
    ):
        parsed = megacpp.parse_metadata_buffer(buf, buffer_len, filename)
        return MegaTensorsMetadata(parsed, buffer_len, framework, filename)

    @classmethod
    def from_fd(
        self,
        fd: int,
        filename: str,
        framework: FrameworkOpBase,
        keep_orig_dict: bool = False,
    ):
        status = os.fstat(fd)
        buffer_len = status.st_size
        if buffer_len < 8:
            raise Exception(f"{filename}: HeaderTooSmall, buffer_len={buffer_len}")
        parsed = megacpp.parse_metadata_fd(fd, filename, buffer_len)
        return MegaTensorsMetadata(
            parsed,
            buffer_len,
            framework,
            filename,
            keep_orig_dict=keep_orig_dict,
        )

    @classmethod
    def from_file(self, filename: str, framework: FrameworkOpBase):
        flags = os.O_RDONLY
        # On Windows, O_RDONLY defaults to text mode which translates \r\n -> \n,
        # corrupting binary data and causing size mismatches on large files.
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(filename, flags, 0o644)
        try:
            return self.from_fd(fd, filename, framework=framework, keep_orig_dict=False)
        finally:
            os.close(fd)

    def get_tensors(
        self,
        gbuf: megacpp.gds_device_buffer,
        device: Device,
        copy_start_offset: int,
        dtype: DType = DType.AUTO,
        keep_tensor: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, TensorBase]:
        if keep_tensor is None and not self.has_encoded_tensors:
            return self._get_raw_dense_tensors_fast(gbuf, device, copy_start_offset, dtype)
        ret = {}
        for tensor_name, t in self.tensors.items():
            if keep_tensor is not None and not keep_tensor(tensor_name):
                continue
            if t.storage_format not in (
                MEGA_RAW_DENSE_STORAGE_FORMAT,
                MEGA_CHUNKED_RAW_DENSE_STORAGE_FORMAT,
            ):
                raise NotImplementedError(
                    f"{self.src}: tensor {tensor_name} uses storage_format={t.storage_format}; "
                    f"raw copier path only supports {MEGA_RAW_DENSE_STORAGE_FORMAT} "
                    f"and {MEGA_CHUNKED_RAW_DENSE_STORAGE_FORMAT}"
                )
            if (
                t.storage_format == MEGA_CHUNKED_RAW_DENSE_STORAGE_FORMAT
                or t.tensor_flags != 0
                or t.compression_codec != 0
            ):
                ret[tensor_name] = self._load_encoded_raw_dense_tensor(
                    tensor_name, t, device, dtype
                )
                continue
            dst_dev_ptr = (
                gbuf.get_base_address()
                + self.header_length
                + t.data_offsets[0]
                - copy_start_offset
            )
            disk_dtype = self.framework.as_workaround_dtype(t.dtype)
            dl_shape, dl_strides = self.framework.get_storage_shape(
                t.dtype, t.shape, t.strides
            )
            dl_tensor = from_cuda_buffer(
                dst_dev_ptr,
                dl_shape,
                dl_strides,
                disk_dtype,
                device,
            )
            t2 = self.framework.from_dlpack(dl_tensor, device, disk_dtype)
            if disk_dtype != t.dtype:
                t2 = t2.view(t.dtype)
            # For packed sub-byte dtypes, reshape to the framework-native shape.
            # MEGA stores logical element shapes; some frameworks expose packed
            # storage units for sub-byte dtypes.
            native_shape = self.framework.get_native_shape(t.dtype, t.shape)
            if native_shape != t.shape:
                t2 = t2.reshape(native_shape)

            if dtype != DType.AUTO and dtype != t.dtype:
                if self.framework.get_dtype_size(dtype) > self.framework.get_dtype_size(
                    t.dtype
                ):
                    raise Exception(
                        f"Online type conversion to larger sizes is not supported ({t.dtype} -> {dtype})"
                    )
                t3 = t2.to(dtype=dtype)
                conv_dtype: DType = self.framework.as_workaround_dtype(dtype)
                dl_tensor = from_cuda_buffer(
                    dst_dev_ptr,
                    t.shape,
                    t.strides,
                    conv_dtype,
                    device,
                )
                t2 = self.framework.from_dlpack(dl_tensor, device, conv_dtype)
                if dtype != conv_dtype:
                    t2 = t2.view(dtype)
                self.framework.copy_tensor(t2, t3)
                self.tensors[tensor_name].dtype = dtype
            ret[tensor_name] = t2
        return ret

    def _get_raw_dense_tensors_fast(
        self,
        gbuf: megacpp.gds_device_buffer,
        device: Device,
        copy_start_offset: int,
        dtype: DType = DType.AUTO,
    ) -> Dict[str, TensorBase]:
        ret = {}
        base_addr = gbuf.get_base_address() + self.header_length - copy_start_offset
        for tensor_name, t in self.tensors.items():
            dst_dev_ptr = base_addr + t.data_offsets[0]
            disk_dtype = self.framework.as_workaround_dtype(t.dtype)
            dl_shape, dl_strides = self.framework.get_storage_shape(
                t.dtype, t.shape, t.strides
            )
            dl_tensor = from_cuda_buffer(
                dst_dev_ptr,
                dl_shape,
                dl_strides,
                disk_dtype,
                device,
            )
            t2 = self.framework.from_dlpack(dl_tensor, device, disk_dtype)
            if disk_dtype != t.dtype:
                t2 = t2.view(t.dtype)
            native_shape = self.framework.get_native_shape(t.dtype, t.shape)
            if native_shape != t.shape:
                t2 = t2.reshape(native_shape)

            if dtype != DType.AUTO and dtype != t.dtype:
                if self.framework.get_dtype_size(dtype) > self.framework.get_dtype_size(
                    t.dtype
                ):
                    raise Exception(
                        f"Online type conversion to larger sizes is not supported ({t.dtype} -> {dtype})"
                    )
                t3 = t2.to(dtype=dtype)
                conv_dtype: DType = self.framework.as_workaround_dtype(dtype)
                dl_tensor = from_cuda_buffer(
                    dst_dev_ptr,
                    t.shape,
                    t.strides,
                    conv_dtype,
                    device,
                )
                t2 = self.framework.from_dlpack(dl_tensor, device, conv_dtype)
                if dtype != conv_dtype:
                    t2 = t2.view(dtype)
                self.framework.copy_tensor(t2, t3)
                self.tensors[tensor_name].dtype = dtype
            ret[tensor_name] = t2
        return ret

    def select_byte_ranges(
        self, keep_tensor: Callable[[str], bool], merge_gap: int = 4096
    ) -> List[Tuple[int, int]]:
        """Compute the file byte-ranges covering only the kept tensors.

        Returns a sorted list of ``[start, end)`` absolute file offsets spanning
        exactly the tensors for which ``keep_tensor(name)`` is True. Kept tensors
        separated by a gap of at most ``merge_gap`` bytes are coalesced into one
        range to reduce the number of reads; the few non-kept bytes inside a
        coalesced range are read but never instantiated as tensors.

        Pass the result to a partial-read-capable copier (see
        ``NoGdsFileCopier.set_byte_ranges``) to load only a subset of a shard --
        e.g. only the experts an expert-parallel rank owns. Tensor data offsets
        are unchanged, so unread regions of the device buffer simply stay
        uninitialized and their tensors must not be requested.
        """
        ranges: List[Tuple[int, int]] = []
        for name, frame in self.tensors.items():
            if not keep_tensor(name):
                continue
            s, e = frame.stored_offsets[0], frame.stored_offsets[1]
            if frame.storage_format == MEGA_CHUNKED_RAW_DENSE_STORAGE_FORMAT:
                for chunk in frame.chunks:
                    start = int(chunk["payload_offset"])
                    end = start + int(chunk["stored_size"])
                    if start != end:
                        ranges.append((self.header_length + start, self.header_length + end))
            elif s != e:
                ranges.append((self.header_length + s, self.header_length + e))
        ranges.sort()
        merged: List[List[int]] = []
        for s, e in ranges:
            if merged and s - merged[-1][1] <= merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    def select_segment_byte_ranges(
        self,
        keep_segment: Callable[[Dict], bool],
        merge_gap: int = 4096,
    ) -> List[Tuple[int, int]]:
        """Compute absolute file byte-ranges for selected segment payloads.

        Segment ranges are execution hints: using them can drive prefetch or
        staged loading, but ignoring them must not change tensor values.
        """
        ranges: List[Tuple[int, int]] = []
        for segment in self.segments:
            if not keep_segment(segment):
                continue
            size = int(segment.get("payload_size", 0))
            if size == 0:
                continue
            start = int(segment["payload_offset"])
            ranges.append((self.header_length + start, self.header_length + start + size))
        ranges.sort()
        merged: List[List[int]] = []
        for s, e in ranges:
            if merged and s - merged[-1][1] <= merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    def get_moe_expert_tensors(
        self,
        layer_id: Optional[int] = None,
        expert_id: Optional[int] = None,
    ) -> List[str]:
        """Return tensor names from the advisory MoE expert table."""
        names: List[str] = []
        for expert in self.moe_experts:
            if layer_id is not None and int(expert["layer_id"]) != layer_id:
                continue
            if expert_id is not None and int(expert["expert_id"]) != expert_id:
                continue
            names.append(str(expert["tensor_name"]))
        return names

    def get_model_info(self) -> MegaModelInfo:
        """Return normalized model-level metadata carried by the MEGA artifact."""
        metadata = self.metadata
        architecture = str(metadata.get("general.architecture", ""))
        if not architecture:
            raise ValueError(f"{self.src}: missing general.architecture")

        def _int(name: str, default: int = 0) -> int:
            return int(metadata.get(name, default))

        return MegaModelInfo(
            architecture=architecture,
            name=str(metadata.get("general.name", "")),
            basename=str(metadata.get("general.basename", "")),
            size_label=str(metadata.get("general.size_label", "")),
            context_length=_int(f"{architecture}.context_length"),
            embedding_length=_int(f"{architecture}.embedding_length"),
            block_count=_int(f"{architecture}.block_count"),
            feed_forward_length=_int(f"{architecture}.feed_forward_length"),
            attention_head_count=_int(f"{architecture}.attention.head_count"),
            attention_head_count_kv=_int(f"{architecture}.attention.head_count_kv"),
            rope_dimension_count=_int(f"{architecture}.rope.dimension_count"),
            vocab_size=_int(f"{architecture}.vocab_size"),
        )

    def get_tokenizer_info(self) -> MegaTokenizerInfo:
        """Return tokenizer metadata embedded in the MEGA artifact."""
        metadata = self.metadata
        model = str(metadata.get("tokenizer.model", ""))
        tokens = [str(token) for token in metadata.get("tokenizer.tokens", [])]
        if not model:
            raise ValueError(f"{self.src}: missing tokenizer.model")
        if not tokens:
            raise ValueError(f"{self.src}: missing tokenizer.tokens")
        token_types = [int(v) for v in metadata.get("tokenizer.token_type", [])]
        if token_types and len(token_types) != len(tokens):
            raise ValueError(
                f"{self.src}: tokenizer.token_type length does not match tokens"
            )
        merges = [str(v) for v in metadata.get("tokenizer.merges", [])]

        def _int(name: str, default: int = -1) -> int:
            return int(metadata.get(name, default))

        return MegaTokenizerInfo(
            model=model,
            tokens=tokens,
            token_types=token_types,
            merges=merges,
            bos_token_id=_int("tokenizer.bos_token_id"),
            eos_token_id=_int("tokenizer.eos_token_id"),
            unknown_token_id=_int("tokenizer.unknown_token_id"),
            padding_token_id=_int("tokenizer.padding_token_id"),
            bos_token=str(metadata.get("tokenizer.bos_token", "")),
            eos_token=str(metadata.get("tokenizer.eos_token", "")),
            unknown_token=str(metadata.get("tokenizer.unknown_token", "")),
            padding_token=str(metadata.get("tokenizer.padding_token", "")),
            pre_tokenizer=str(metadata.get("tokenizer.pre_tokenizer", "whitespace")),
            add_bos_token=bool(metadata.get("tokenizer.add_bos_token", False)),
            add_eos_token=bool(metadata.get("tokenizer.add_eos_token", False)),
        )

    def get_hf_tokenizer(self):
        """Construct a Hugging Face tokenizer object directly from typed MEGA KV."""
        info = self.get_tokenizer_info()
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE, WordLevel
            from tokenizers.pre_tokenizers import ByteLevel, Whitespace
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            raise ImportError(
                "get_hf_tokenizer requires the tokenizers and transformers packages"
            ) from exc

        vocab = {token: idx for idx, token in enumerate(info.tokens)}
        if info.model == "wordlevel":
            model = WordLevel(vocab, unk_token=info.unknown_token or None)
        elif info.model == "bpe":
            merges = [tuple(merge.split()) for merge in info.merges]
            model = BPE(vocab=vocab, merges=merges, unk_token=info.unknown_token or None)
        else:
            raise ValueError(f"{self.src}: unsupported tokenizer.model={info.model}")
        tokenizer = Tokenizer(model)
        if info.pre_tokenizer == "bytelevel":
            tokenizer.pre_tokenizer = ByteLevel()
        elif info.pre_tokenizer == "whitespace":
            tokenizer.pre_tokenizer = Whitespace()
        elif info.pre_tokenizer:
            raise ValueError(f"{self.src}: unsupported tokenizer.pre_tokenizer={info.pre_tokenizer}")
        kwargs = {}
        if info.bos_token:
            kwargs["bos_token"] = info.bos_token
        if info.eos_token:
            kwargs["eos_token"] = info.eos_token
        if info.unknown_token:
            kwargs["unk_token"] = info.unknown_token
        if info.padding_token:
            kwargs["pad_token"] = info.padding_token
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer, **kwargs)

    def compute_payload_crc32(self) -> int:
        """Compute CRC32 over the payload data section on demand."""
        flags = os.O_RDONLY
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(self.src, flags, 0o644)
        try:
            return int(
                megacpp.payload_crc32_fd(fd, self.src, self.header_length, self.size_bytes)
            )
        finally:
            os.close(fd)

    def verify_payload_crc32(self, expected: Optional[int] = None) -> bool:
        """Verify ``mega.hash.payload.crc32`` or an explicit expected CRC32.

        This scans the payload section and is intentionally explicit so normal
        metadata open and raw mmap paths remain zero-copy.
        """
        if expected is None:
            expected = self.metadata.get("mega.hash.payload.crc32")
        if expected is None:
            raise ValueError(f"{self.src}: missing mega.hash.payload.crc32")
        if isinstance(expected, str):
            expected = int(expected, 16 if expected.lower().startswith("0x") else 10)
        expected = int(expected) & 0xFFFFFFFF
        actual = self.compute_payload_crc32()
        if actual != expected:
            raise ValueError(
                f"{self.src}: payload CRC32 mismatch, expected=0x{expected:08x}, actual=0x{actual:08x}"
            )
        return True

    def compute_payload_sha256(self) -> str:
        """Compute SHA-256 over the payload data section on demand."""
        flags = os.O_RDONLY
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(self.src, flags, 0o644)
        try:
            digest = megacpp.payload_sha256_fd(
                fd, self.src, self.header_length, self.size_bytes
            )
        finally:
            os.close(fd)
        return bytes(digest).hex()

    def verify_payload_sha256(self, expected: Optional[str] = None) -> bool:
        """Verify ``mega.hash.payload.sha256`` or an explicit expected digest."""
        if expected is None:
            expected = self.metadata.get("mega.hash.payload.sha256")
        if expected is None:
            raise ValueError(f"{self.src}: missing mega.hash.payload.sha256")
        expected = str(expected).strip().lower()
        actual = self.compute_payload_sha256()
        if actual != expected:
            raise ValueError(
                f"{self.src}: payload SHA256 mismatch, expected={expected}, actual={actual}"
            )
        return True

    def _trust_result(
        self,
        risk: str,
        *,
        strict: bool,
        warn: bool,
        **fields,
    ) -> Dict:
        result = {
            "trusted": risk == "trusted",
            "risk": risk,
            "source_risk": risk != "trusted",
            **fields,
        }
        if risk != "trusted":
            message = f"{self.src}: MEGA artifact source risk: {risk}"
            if strict:
                raise ValueError(message)
            if warn:
                warnings.warn(message, RuntimeWarning, stacklevel=3)
        return result

    @staticmethod
    def _parse_trust_statement(statement: bytes) -> Dict[str, str]:
        try:
            text = statement.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("statement is not valid UTF-8")
        if not text.startswith(MEGA_TRUST_SIGNATURE_STATEMENT_PREFIX):
            raise ValueError("statement missing MEGA-TRUST-v1 prefix")
        if not text.endswith("\n"):
            raise ValueError("statement must end with a newline")
        lines = text.splitlines()
        body = lines[1:]
        expected_len = len(MEGA_TRUST_STATEMENT_KEYS)
        if len(body) != expected_len:
            raise ValueError(
                f"statement must contain exactly {expected_len} canonical fields"
            )
        values: Dict[str, str] = {}
        for line, key in zip(body, MEGA_TRUST_STATEMENT_KEYS):
            prefix = f"{key}="
            if not line.startswith(prefix):
                raise ValueError(f"statement field order violation at {key}")
            value = line[len(prefix) :]
            if value == "":
                raise ValueError(f"statement field {key} is empty")
            if "\r" in value or "\n" in value:
                raise ValueError(f"statement field {key} contains a newline")
            values[key] = value
        if values["format_version"] != "1":
            raise ValueError("unsupported statement format_version")
        if values["artifact_kind"] != "model":
            raise ValueError("statement artifact_kind must be model")
        for key in ("created_at", "expires_at"):
            try:
                int(values[key])
            except ValueError as exc:
                raise ValueError(f"statement {key} must be a Unix timestamp") from exc
        return values

    @staticmethod
    def _allowed(value: str, allowed: Optional[Iterable[str]]) -> bool:
        if allowed is None:
            return True
        return value in set(allowed)

    def verify_trust(
        self,
        trusted_roots_pem: str,
        *,
        strict: bool = False,
        warn: bool = True,
        allowed_model_ids: Optional[Iterable[str]] = None,
        now: Optional[int] = None,
    ) -> Dict:
        """Verify artifact provenance using an external X.509 trust root.

        The artifact may carry a leaf certificate, optional intermediate chain,
        signed statement, and signature. Trust roots must come from caller
        policy, not from the artifact itself.
        """
        leaf_pem = self.metadata.get("mega.trust.certificate.leaf_pem")
        statement = self.metadata.get("mega.trust.signature.statement")
        signature_b64 = self.metadata.get("mega.trust.signature.value")
        algorithm = self.metadata.get("mega.trust.signature.algorithm")
        cert_format = self.metadata.get("mega.trust.certificate.format")
        if not leaf_pem or not statement or not signature_b64 or not algorithm:
            return self._trust_result("unsigned", strict=strict, warn=warn)
        if cert_format != MEGA_TRUST_CERTIFICATE_FORMAT:
            return self._trust_result(
                "unsupported_certificate_format",
                strict=strict,
                warn=warn,
                certificate_format=cert_format,
            )
        if not trusted_roots_pem:
            return self._trust_result("untrusted_issuer", strict=strict, warn=warn)

        try:
            signature = base64.b64decode(str(signature_b64), validate=True)
        except Exception as exc:
            return self._trust_result(
                "signature_invalid",
                strict=strict,
                warn=warn,
                signature_error=str(exc),
            )

        statement_bytes = str(statement).encode("utf-8")
        try:
            statement_fields = self._parse_trust_statement(statement_bytes)
        except ValueError as exc:
            return self._trust_result(
                "statement_invalid",
                strict=strict,
                warn=warn,
                statement_error=str(exc),
            )

        payload_sha256 = self.compute_payload_sha256()
        metadata_sha256 = self.metadata.get("mega.hash.payload.sha256")
        if metadata_sha256 is not None and str(metadata_sha256).strip().lower() != payload_sha256:
            return self._trust_result(
                "hash_mismatch",
                strict=strict,
                warn=warn,
                payload_sha256=payload_sha256,
                declared_payload_sha256=str(metadata_sha256).strip().lower(),
            )
        statement_sha256 = statement_fields["payload_sha256"].strip().lower()
        if statement_sha256 != payload_sha256:
            return self._trust_result(
                "hash_mismatch",
                strict=strict,
                warn=warn,
                payload_sha256=payload_sha256,
                statement_payload_sha256=statement_sha256,
            )
        if not self._allowed(statement_fields["model_id"], allowed_model_ids):
            return self._trust_result(
                "model_not_allowed",
                strict=strict,
                warn=warn,
                model_id=statement_fields["model_id"],
            )
        current_time = int(time.time()) if now is None else int(now)
        created_at = int(statement_fields["created_at"])
        expires_at = int(statement_fields["expires_at"])
        if created_at > current_time:
            return self._trust_result(
                "statement_not_yet_valid",
                strict=strict,
                warn=warn,
                created_at=created_at,
                now=current_time,
            )
        if expires_at <= current_time:
            return self._trust_result(
                "statement_expired",
                strict=strict,
                warn=warn,
                expires_at=expires_at,
                now=current_time,
            )

        result = megacpp.verify_x509_signature(
            str(leaf_pem),
            str(self.metadata.get("mega.trust.certificate.chain_pem", "")),
            trusted_roots_pem,
            statement_bytes,
            signature,
            str(algorithm),
        )
        result = dict(result)
        result["statement"] = statement_fields
        result["payload_sha256"] = payload_sha256
        result["source_risk"] = not bool(result.get("trusted"))
        subject = str(result.get("subject", ""))
        if bool(result.get("trusted")) and statement_fields["publisher"] != subject:
            result["trusted"] = False
            result["source_risk"] = True
            result["risk"] = "publisher_certificate_mismatch"
        risk = str(result.get("risk", "untrusted_issuer"))
        if risk != "trusted":
            message = f"{self.src}: MEGA artifact source risk: {risk}"
            if strict:
                raise ValueError(message)
            if warn:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        return result

    def __repr__(self) -> str:
        return str(
            {
                "metadata": self.metadata,
                "tensors": self.tensors,
                "segments": self.segments,
                "moe_experts": self.moe_experts,
            }
        )


class MegaKvMetadata:
    def __init__(
        self,
        parsed: Dict,
        size_bytes: int,
        framework: FrameworkOpBase,
        src: str = "",
    ):
        self.src = src
        self.framework = framework
        self.version = int(parsed["version"])
        self.metadata = dict(parsed["metadata"])
        self.payload_offset = int(parsed["payload_offset"])
        self.entries: List[MegaKvEntry] = [
            self._entry_from_dict(dict(entry)) for entry in parsed["entries"]
        ]
        self.size_bytes = size_bytes
        self.entries_by_name = {entry.name: entry for entry in self.entries}
        if len(self.entries_by_name) != len(self.entries):
            raise ValueError(f"{src}: duplicate MEGAKV entry name")
        if self.payload_offset > size_bytes:
            raise ValueError(f"{src}: MEGAKV payload offset exceeds file size")

    @staticmethod
    def _entry_from_dict(entry: Dict) -> MegaKvEntry:
        return MegaKvEntry(
            entry_id=int(entry["entry_id"]),
            name=str(entry["name"]),
            layer_id=int(entry["layer_id"]),
            kv_role=int(entry["kv_role"]),
            sequence_id=int(entry["sequence_id"]),
            token_start=int(entry["token_start"]),
            token_count=int(entry["token_count"]),
            shape=[int(v) for v in entry["shape"]],
            dtype=DType(entry["dtype"]),
            payload_offset=int(entry["payload_offset"]),
            logical_nbytes=int(entry["logical_nbytes"]),
            stored_nbytes=int(entry["stored_nbytes"]),
            flags=int(entry["flags"]),
            codec=int(entry["codec"]),
            shuffle_elem_size=int(entry["shuffle_elem_size"]),
            checksum_type=int(entry["checksum_type"]),
            checksum=bytes(entry.get("checksum", b"\0" * 32)),
        )

    @classmethod
    def from_fd(self, fd: int, filename: str, framework: FrameworkOpBase):
        status = os.fstat(fd)
        if status.st_size < 24:
            raise ValueError(f"{filename}: MEGAKV header too small")
        parsed = megacpp.parse_kv_fd(fd, filename, status.st_size)
        return MegaKvMetadata(parsed, status.st_size, framework, filename)

    @classmethod
    def from_file(self, filename: str, framework: FrameworkOpBase):
        flags = os.O_RDONLY
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(filename, flags, 0o644)
        try:
            return self.from_fd(fd, filename, framework=framework)
        finally:
            os.close(fd)

    def select_entries(
        self,
        *,
        layer_id: Optional[int] = None,
        kv_role: Optional[int] = None,
        sequence_id: Optional[int] = None,
    ) -> List[MegaKvEntry]:
        entries = []
        for entry in self.entries:
            if layer_id is not None and entry.layer_id != layer_id:
                continue
            if kv_role is not None and entry.kv_role != kv_role:
                continue
            if sequence_id is not None and entry.sequence_id != sequence_id:
                continue
            entries.append(entry)
        return entries

    def load_entry(
        self,
        name: str,
        device: Device,
        dtype: DType = DType.AUTO,
    ) -> TensorBase:
        entry = self.entries_by_name[name]
        tensor = self.framework.get_empty_tensor(entry.shape, entry.dtype, device)
        flags = os.O_RDONLY
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(self.src, flags, 0o644)
        try:
            megacpp.decode_kv_entry_fd(
                fd,
                self.src,
                entry.name,
                self.payload_offset + entry.payload_offset,
                entry.stored_nbytes,
                entry.logical_nbytes,
                entry.flags,
                entry.codec,
                entry.shuffle_elem_size,
                entry.checksum_type,
                entry.checksum,
                tensor.data_ptr(),
                device.type != DeviceType.CPU,
            )
        finally:
            os.close(fd)
        if dtype != DType.AUTO and dtype != entry.dtype:
            if self.framework.get_dtype_size(dtype) > self.framework.get_dtype_size(
                entry.dtype
            ):
                raise Exception(
                    f"Online type conversion to larger sizes is not supported ({entry.dtype} -> {dtype})"
                )
            tensor = tensor.to(dtype=dtype)
        return tensor


@dataclass
class TensorFrame:
    dtype: DType
    shape: List[int]
    data_offsets: List[int]
    strides: List[int]
    offsets: List[int]
    sliced: bool
    storage_format: str = MEGA_RAW_DENSE_STORAGE_FORMAT
    stored_offsets: List[int] = None
    logical_nbytes: int = 0
    stored_nbytes: int = 0
    tensor_flags: int = 0
    compression_codec: int = 0
    shuffle_elem_size: int = 0
    chunks: List[Dict] = None
    tensor_id: int = -1
    checksum_type: int = 0
    checksum: bytes = b"\0" * 32

    def __post_init__(self) -> None:
        if self.stored_offsets is None:
            self.stored_offsets = self.data_offsets
        if self.logical_nbytes == 0:
            self.logical_nbytes = self.data_offsets[1] - self.data_offsets[0]
        if self.stored_nbytes == 0:
            self.stored_nbytes = self.stored_offsets[1] - self.stored_offsets[0]
        if self.chunks is None:
            self.chunks = []
        if len(self.checksum) != 32:
            raise ValueError("TensorFrame checksum must be 32 bytes")

    @classmethod
    def from_buffer(self, entry: OrderedDict[str, List[int]]):
        shape = entry["shape"]
        data_offsets = list(entry["data_offsets"])
        stored_offsets = list(entry.get("stored_offsets", data_offsets))
        strides = [1] * len(shape)
        running = 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = running
            running *= shape[i]
        offsets = [0] * len(shape)
        return TensorFrame(
            DType(entry["dtype"]),
            shape,
            data_offsets,
            strides,
            offsets,
            False,
            str(entry.get("storage_format", MEGA_RAW_DENSE_STORAGE_FORMAT)),
            stored_offsets,
            int(entry.get("logical_nbytes", data_offsets[1] - data_offsets[0])),
            int(entry.get("stored_nbytes", stored_offsets[1] - stored_offsets[0])),
            int(entry.get("tensor_flags", 0)),
            int(entry.get("compression_codec", 0)),
            int(entry.get("shuffle_elem_size", 0)),
            list(entry.get("chunks", [])),
            int(entry.get("tensor_id", -1)),
            int(entry.get("checksum_type", 0)),
            bytes(entry.get("checksum", b"\0" * 32)),
        )

    @classmethod
    def from_record(cls, record):
        (
            name,
            tensor_id,
            dtype,
            shape,
            payload_offset,
            logical_nbytes,
            stored_nbytes,
            tensor_flags,
            compression_codec,
            shuffle_elem_size,
            checksum_type,
            checksum,
            storage_format,
            chunks,
        ) = record
        shape = list(shape)
        strides = [1] * len(shape)
        running = 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = running
            running *= shape[i]
        data_offsets = [int(payload_offset), int(payload_offset) + int(logical_nbytes)]
        stored_offsets = [int(payload_offset), int(payload_offset) + int(stored_nbytes)]
        py_chunks = [
            {
                "tensor_id": int(chunk[0]),
                "chunk_id": int(chunk[1]),
                "logical_offset": int(chunk[2]),
                "logical_size": int(chunk[3]),
                "payload_offset": int(chunk[4]),
                "stored_size": int(chunk[5]),
                "codec": int(chunk[6]),
                "flags": int(chunk[7]),
                "checksum_type": int(chunk[8]),
                "checksum": bytes(chunk[9]),
            }
            for chunk in chunks
        ]
        return str(name), TensorFrame(
            DType(dtype),
            shape,
            data_offsets,
            strides,
            [0] * len(shape),
            False,
            str(storage_format),
            stored_offsets,
            int(logical_nbytes),
            int(stored_nbytes),
            int(tensor_flags),
            int(compression_codec),
            int(shuffle_elem_size),
            py_chunks,
            int(tensor_id),
            int(checksum_type),
            bytes(checksum),
        )

    def __repr__(self) -> str:
        return str(
            {
                "dtype": self.dtype,
                "shape": self.shape,
                "data_offsets": self.data_offsets,
                "storage_format": self.storage_format,
                "stored_offsets": self.stored_offsets,
                "tensor_flags": self.tensor_flags,
                "compression_codec": self.compression_codec,
                "chunks": self.chunks,
            }
        )

    # TODO: reduce dim if isinstance(_val, int) == True
    def __getitem__(self, _val) -> "TensorFrame":
        val: Tuple = ()
        if isinstance(_val, slice) or isinstance(_val, int):
            val = (_val,)
        elif isinstance(_val, tuple):
            val = _val
        else:
            raise Exception(f"[BUG] Unsupported index type for DiskTensor: {_val}")
        if len(val) > len(self.shape):
            raise Exception(
                f"[BUG] tried to get too large slice {_val} from {self.shape}"
            )
        shape: List[int] = []
        strides: List[int] = []
        offsets: List[int] = []
        for dim in range(0, len(val)):
            if isinstance(val[dim], int):
                start = val[dim]
                if start >= self.shape[dim] or start < -self.shape[dim]:
                    raise IndexError(
                        f"[BUG] tried to access index {start} at dim={dim} for shape={self.shape}"
                    )
                if start < 0:
                    start = self.shape[dim] + start
                step = 1
                length = 1
            elif isinstance(val[dim], slice):
                if val[dim].step == 0:
                    raise ValueError(f"[BUG] slice step cannot be zero")
                # normalize None/negative/out-of-range bounds the same way
                # Python sequences do
                start, stop, step = val[dim].indices(self.shape[dim])
                length = stop - start
                if (
                    length == 0
                    or (length < 0 and step > 0)
                    or (length > 0 and step < 0)
                ):
                    return TensorFrame(
                        self.dtype,
                        [],
                        self.data_offsets,
                        [],
                        [],
                        False,
                        self.storage_format,
                        self.stored_offsets,
                        self.logical_nbytes,
                        self.stored_nbytes,
                        self.tensor_flags,
                        self.compression_codec,
                        self.shuffle_elem_size,
                        self.chunks,
                        self.tensor_id,
                        self.checksum_type,
                        self.checksum,
                    )
                if length < 0 and step < 0:
                    length *= -1
            else:
                raise Exception(
                    f"[BUG] Unsupported index type for DiskTensor: {_val} at dim={dim}"
                )
            offsets.append(self.offsets[dim] + start)
            strides.append(self.strides[dim] * step)
            abs_step = step if step > 0 else -step
            shape.append((length + abs_step - 1) // abs_step)
        for rdim in range(len(val), len(self.shape)):
            offsets.append(self.offsets[rdim])
            strides.append(self.strides[rdim])
            shape.append(self.shape[rdim])
        return TensorFrame(
            self.dtype,
            shape,
            self.data_offsets,
            strides,
            offsets,
            True,
            self.storage_format,
            self.stored_offsets,
            self.logical_nbytes,
            self.stored_nbytes,
            self.tensor_flags,
            self.compression_codec,
            self.shuffle_elem_size,
            self.chunks,
            self.tensor_id,
            self.checksum_type,
            self.checksum,
        )
