# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Union

from . import cpp as megacpp
from .common import MegaKvMetadata, MegaTensorsMetadata
from .convert import read_index_metadata, resolve_artifacts
from .frameworks import get_framework_op
from .loader import mega_open
from .st_types import DType


PathLike = Union[str, Iterable[str], Dict[int, list[str]]]


def _close_owner(owner: Any) -> None:
    if hasattr(owner, "close"):
        owner.close()
    elif hasattr(owner, "__exit__"):
        owner.__exit__(None, None, None)


class BorrowedStateDict(OrderedDict):
    """State dict whose tensors borrow storage from an open MEGA artifact."""

    def __init__(self, owner: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mega_owner = owner

    def close(self) -> None:
        owner = getattr(self, "_mega_owner", None)
        if owner is not None:
            _close_owner(owner)
            self._mega_owner = None


def _clone_tensor(tensor: Any) -> Any:
    if hasattr(tensor, "clone"):
        return tensor.clone()
    return tensor


def iter_tensors(
    filenames: PathLike,
    *,
    device: str = "cpu",
    dtype: DType = DType.AUTO,
    tensor_filter: Optional[Callable[[str], bool]] = None,
    key_mapping: Optional[Callable[[str], str]] = None,
    framework: str = "pt",
    nogds: bool = False,
) -> Iterator[tuple[str, Any]]:
    """Yield cloned ``(name, tensor)`` pairs outside the file-buffer lifetime."""
    with mega_open(
        filenames,
        framework=framework,
        device=device,
        nogds=nogds,
    ) as artifact:
        for name in artifact.keys():
            if tensor_filter is not None and not tensor_filter(name):
                continue
            out_name = key_mapping(name) if key_mapping is not None else name
            yield out_name, _clone_tensor(artifact.fb.get_tensor(name, dtype=dtype))


def load_tensor(
    filenames: PathLike,
    name: str,
    *,
    device: str = "cpu",
    dtype: DType = DType.AUTO,
    framework: str = "pt",
    nogds: bool = False,
) -> Any:
    """Load one MEGA tensor by name and clone it out of the file buffer."""
    with mega_open(
        filenames,
        framework=framework,
        device=device,
        nogds=nogds,
    ) as artifact:
        return _clone_tensor(artifact.fb.get_tensor(name, dtype=dtype))


def load_state_dict(
    filenames: PathLike,
    *,
    device: str = "cpu",
    dtype: DType = DType.AUTO,
    tensor_filter: Optional[Callable[[str], bool]] = None,
    key_mapping: Optional[Callable[[str], str]] = None,
    framework: str = "pt",
    nogds: bool = False,
    strict_unique: bool = True,
    borrow: bool = False,
) -> "OrderedDict[str, Any]":
    """Load MEGA weights as a PyTorch-style ``state_dict``."""
    if borrow:
        artifact = mega_open(
            filenames,
            framework=framework,
            device=device,
            nogds=nogds,
        )
        state_dict: "OrderedDict[str, Any]" = BorrowedStateDict(artifact)
        try:
            for name in artifact.keys():
                if tensor_filter is not None and not tensor_filter(name):
                    continue
                out_name = key_mapping(name) if key_mapping is not None else name
                if strict_unique and out_name in state_dict:
                    raise ValueError(f"duplicate state_dict key after mapping: {out_name}")
                state_dict[out_name] = artifact.fb.get_tensor(name, dtype=dtype)
        except Exception:
            _close_owner(artifact)
            raise
        return state_dict
    state_dict: "OrderedDict[str, Any]" = OrderedDict()
    for out_name, tensor in iter_tensors(
        filenames,
        device=device,
        dtype=dtype,
        tensor_filter=tensor_filter,
        key_mapping=key_mapping,
        framework=framework,
        nogds=nogds,
    ):
        if strict_unique and out_name in state_dict:
            raise ValueError(f"duplicate state_dict key after mapping: {out_name}")
        state_dict[out_name] = tensor
    return state_dict


def load_model(
    filenames: PathLike,
    *,
    device: str = "cpu",
    dtype: DType = DType.AUTO,
    tensor_filter: Optional[Callable[[str], bool]] = None,
    key_mapping: Optional[Callable[[str], str]] = None,
    model_class: Optional[Union[str, Callable[..., Any]]] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
    assign: bool = False,
    nogds: bool = False,
) -> Any:
    """Construct a model object from MEGA metadata and load its weights."""
    metadata_filename = filenames if isinstance(filenames, str) else _first_filename(filenames)
    fw = get_framework_op("pt")
    if str(metadata_filename).endswith(".mega.index.json"):
        metadata = read_index_metadata(metadata_filename)
    else:
        metadata = MegaTensorsMetadata.from_file(metadata_filename, fw).metadata
    cls_spec = model_class or metadata.get("model.class")
    if cls_spec is None:
        raise ValueError(
            f"{metadata_filename}: missing model.class metadata; pass model_class"
        )
    cls = _resolve_model_class(cls_spec)
    init_kwargs = _model_init_kwargs(metadata)
    if model_kwargs is not None:
        init_kwargs.update(dict(model_kwargs))
    model = cls(**init_kwargs)
    prefix = str(metadata.get("model.state_dict.prefix", ""))
    if key_mapping is None and prefix:
        key_mapping = lambda key: key.removeprefix(prefix)
    state_dict = load_state_dict(
        filenames,
        device=device,
        dtype=dtype,
        tensor_filter=tensor_filter,
        key_mapping=key_mapping,
        framework="pt",
        nogds=nogds,
        borrow=assign,
    )
    try:
        model.load_state_dict(state_dict, strict=strict, assign=assign)
    except TypeError:
        model.load_state_dict(state_dict, strict=strict)
    if hasattr(model, "to"):
        model = model.to(device)
    if isinstance(state_dict, BorrowedStateDict):
        setattr(model, "_mega_state_owner", state_dict)
    return model


def _first_filename(filenames: PathLike) -> str:
    if isinstance(filenames, str):
        resolved = resolve_artifacts(filenames)
        if resolved:
            return resolved[0]
        return filenames
    if isinstance(filenames, dict):
        for values in filenames.values():
            if values:
                return values[0]
    else:
        for value in filenames:
            return value
    raise ValueError("filenames is empty")


def _resolve_model_class(
    spec: Union[str, Callable[..., Any]],
) -> Callable[..., Any]:
    if callable(spec):
        return spec
    module_name, _, attr = str(spec).rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"model class must be fully qualified: {spec!r}")
    module = import_module(module_name)
    constructor = getattr(module, attr)
    if not callable(constructor):
        raise ValueError(f"model class must resolve to a callable: {spec!r}")
    return constructor


def _model_init_kwargs(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    prefix = "model.init."
    out: Dict[str, Any] = {}
    for key, value in metadata.items():
        if not str(key).startswith(prefix):
            continue
        name = str(key)[len(prefix) :]
        out[name] = value
    return out


def load_tokenizer(filename: str, *, framework: str = "pt") -> Any:
    """Load the embedded tokenizer as a Hugging Face tokenizer Python object."""
    fw = get_framework_op(framework)
    if str(filename).endswith(".mega.index.json"):
        parsed = {
            "version": 1,
            "metadata": read_index_metadata(filename),
            "tensor_records": [],
            "header_length": 0,
        }
        metadata = MegaTensorsMetadata(parsed, 0, fw, src=filename)
    else:
        metadata = MegaTensorsMetadata.from_file(filename, fw)
    return metadata.get_hf_tokenizer()


def open_kv_cache(filename: str, *, framework: str = "pt") -> MegaKvMetadata:
    """Open a MEGAKV sidecar and expose typed KV-cache entries."""
    fw = get_framework_op(framework)
    return MegaKvMetadata.from_file(filename, fw)


def load_kv_tensor(
    filename: str,
    name: str,
    *,
    device: str = "cpu",
    dtype: DType = DType.AUTO,
    framework: str = "pt",
) -> Any:
    """Load one tensor from a MEGAKV sidecar by entry name."""
    fw = get_framework_op(framework)
    kv = MegaKvMetadata.from_file(filename, fw)
    pg = fw.get_process_group(None)
    return kv.load_entry(name, fw.get_device(device, pg), dtype).get_raw()


def write_kv_cache(
    filename: str,
    entries: Iterable[Mapping[str, Any]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    alignment: int = 32,
) -> None:
    """Write a raw MEGAKV sidecar.

    ``entries`` are dictionaries with ``name``, ``data``, ``shape``,
    ``logical_dtype`` and optional KV identity/checksum fields. This writer is
    intentionally narrow: compression is consumed by the C++ reader but not
    invented here.
    """
    normalized = []
    for entry in entries:
        item = dict(entry)
        data = item["data"]
        if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "numpy"):
            data = data.detach().cpu().numpy().tobytes()
        elif hasattr(data, "tobytes"):
            data = data.tobytes()
        item["data"] = bytes(data)
        normalized.append(item)
    meta = {
        "general.alignment": int(alignment),
        "mega.artifact.kind": "kvcache",
        "megakv.format": "v1",
        **dict(metadata or {}),
    }
    megacpp.write_kv_file(filename, normalized, meta, int(alignment))


def append_footer_overlay(
    filename: str,
    metadata: Mapping[str, Any],
    *,
    generation: int = 1,
) -> None:
    """Append a metadata-only MEGA footer overlay via the C++ backend."""
    megacpp.append_footer_overlay_file(filename, dict(metadata), int(generation))
