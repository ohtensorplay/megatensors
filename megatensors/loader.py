# SPDX-License-Identifier: Apache-2.0

import math
import platform
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from . import cpp as megacpp
from .common import (
    MegaTensorsMetadata,
    TensorFrame,
    get_device_numa_node,
    init_logger,
)
from .copier import CopierConstructFunc, CopierType, create_copier_constructor
from .copier.unified import is_unified_memory_system
from .convert import resolve_artifacts
from .file_buffer import FilesBufferOnDevice
from .frameworks import TensorBase, get_framework_op
from .st_types import Device, DeviceType, DType
from .tensor_factory import LazyTensorFactory

gl_set_numa = False

loaded_library = False

logger = init_logger(__name__)


class BaseMegaTensorsFileLoader:
    r"""Base class for loading .mega files lazily.

    Args:
        pg (Optional[Any]): Process group-like objects for distributed loading.
                           Use None for single device use-cases.
        device (Device): Target device where tensors will be loaded (CPU, CUDA, etc.).
        copier_constructor: Constructor function for creating file copier objects.
        set_numa (bool): Whether to set NUMA node affinity for optimized memory access.
        disable_cache (bool): Whether to disable caching of loaded tensors.
        debug_log (bool): Enable detailed debug logging.
        framework (str): Deep learning framework to use ("pytorch" or "paddle").
    """

    @classmethod
    def process_extension_config(
        cls, ext_config: Mapping[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """Translate extension config into ``__init__`` kwargs.
        Default: shallow copy as-is. Subclasses override to remap fields.
        ``kwargs`` carries runtime context (e.g. ``hf_weights_files``)."""
        return dict(ext_config)

    def __init__(
        self,
        pg: Optional[Any],
        device: Device,
        copier_type: CopierType,
        set_numa: bool = True,
        disable_cache: bool = True,
        framework="pytorch",
        **kwargs,
    ):
        self.framework = get_framework_op(framework)
        self.pg = self.framework.get_process_group(pg)
        self.device = device
        self.meta: Dict[str, Tuple[MegaTensorsMetadata, int]] = {}
        self.frames = OrderedDict[str, TensorFrame]()
        self.disable_cache = disable_cache
        self._tensor_filter: Optional[Callable[[str], bool]] = None
        self._segment_filter: Optional[Callable[[Dict], bool]] = None
        self._segment_merge_gap = 4096
        self._segment_visible_tensor_names: Optional[set[str]] = None
        self.init_numa(set_numa)
        self.copier_constructor: CopierConstructFunc = create_copier_constructor(
            copier_type=copier_type,
            device=device,
            framework=self.framework,
            **kwargs,
        )

    def init_numa(self, set_numa: bool = True):
        global gl_set_numa
        if not gl_set_numa and set_numa:
            node = get_device_numa_node(self.device.index)
            if node is not None:
                megacpp.set_numa_node(node)
            gl_set_numa = True

    def reset(self):
        self.frames = {}
        self.meta = {}

    def close(self):
        self.reset()
        del self.copier_constructor

    def get_keys(self) -> List[str]:
        return [k for k in self.frames.keys() if self._keep_tensor_name(k)]

    def get_shape(self, tensor_name: str) -> List[int]:
        if not self._keep_tensor_name(tensor_name):
            raise ValueError(f"get_shape: key {tensor_name} is filtered out")
        return self.frames[tensor_name].shape

    def _keep_tensor_name(self, tensor_name: str) -> bool:
        if (
            self._segment_filter is not None
            and self._segment_visible_tensor_names is None
        ):
            self._segment_visible_tensor_names = (
                self._compute_segment_visible_tensor_names()
            )
        if self._tensor_filter is not None and not self._tensor_filter(tensor_name):
            return False
        if (
            self._segment_visible_tensor_names is not None
            and tensor_name not in self._segment_visible_tensor_names
        ):
            return False
        return True

    def set_tensor_filter(self, keep_tensor: Optional[Callable[[str], bool]]) -> None:
        """Load only the tensors for which ``keep_tensor(name)`` is True.

        The ``nogds`` and ``unified`` copiers skip reading bytes for filtered
        tensors; other copiers load the full file. The filter narrows the
        public API on every backend: ``get_keys()`` omits filtered tensors,
        ``FilesBufferOnDevice`` does not register them, and ``get_tensor``,
        ``get_filename``, and ``get_shape`` raise ``ValueError`` for them.
        ``ParallelLoader.iterate_weights()`` skips them. ``None`` (the
        default) loads every tensor. See
        ``megatensors.ep_slice.expert_parallel_filter``.
        """
        self._tensor_filter = keep_tensor

    def set_segment_filter(
        self,
        keep_segment: Optional[Callable[[Dict], bool]],
        merge_gap: int = 4096,
    ) -> None:
        """Stage-load only tensors covered by selected segment records.

        Segment records are advisory execution hints. This filter uses selected
        segment payload ranges for range-capable copiers and narrows the public
        tensor API to the tensors whose ``tensor_id`` falls inside the selected
        segment tensor ranges. If ``set_tensor_filter`` is also set, the two
        filters are intersected.
        """
        self._segment_filter = keep_segment
        self._segment_merge_gap = merge_gap
        self._segment_visible_tensor_names = None

    def _compute_segment_visible_tensor_names(self) -> Optional[set[str]]:
        if self._segment_filter is None:
            return None
        visible: set[str] = set()
        for metadata, _ in self.meta.values():
            selected_ids: set[int] = set()
            for segment in metadata.segments:
                if not self._segment_filter(segment):
                    continue
                first = int(segment.get("first_tensor_id", 0))
                count = int(segment.get("tensor_count", 0))
                selected_ids.update(range(first, first + count))
            for name, frame in metadata.tensors.items():
                if frame.tensor_id in selected_ids:
                    visible.add(name)
        return visible

    def _has_tensor_visibility_filter(self) -> bool:
        return self._tensor_filter is not None or self._segment_filter is not None

    def _select_copy_byte_ranges(
        self,
        metadata: MegaTensorsMetadata,
    ) -> Optional[List[Tuple[int, int]]]:
        if self._segment_filter is not None:
            return metadata.select_segment_byte_ranges(
                self._segment_filter,
                merge_gap=self._segment_merge_gap,
            )
        if self._tensor_filter is not None:
            return metadata.select_byte_ranges(self._keep_tensor_name)
        return None

    def add_filenames(self, filenames: Dict[int, List[str]]):
        """
        Register files to ranks to be copied at copy_file_to_device().
        """
        # shuffle files in a round-robin fashion to avoid OoM
        rank_next_idx = {rank: 0 for rank in filenames.keys()}
        completed = 0
        while completed < len(filenames.keys()):
            completed = 0
            for rank in filenames.keys():
                next_idx = rank_next_idx[rank]
                if next_idx < len(filenames[rank]):
                    realpath = filenames[rank][next_idx]  # os.path.realpath(filename)
                    metadata = MegaTensorsMetadata.from_file(realpath, self.framework)
                    self.meta[realpath] = (metadata, rank)
                    self.frames.update(metadata.tensors)
                    self._segment_visible_tensor_names = None
                    if rank == self.pg.rank():
                        logger.debug(
                            "add_filenames %d: path=%s", len(self.meta), realpath
                        )
                    rank_next_idx[rank] = next_idx + 1
                else:
                    completed += 1

    def copy_files_to_device(
        self,
        dtype: DType = DType.AUTO,
        use_buf_register: bool = True,
        max_copy_block_size: int = 256 * 1024 * 1024,
    ) -> FilesBufferOnDevice:
        """
        trigger copying all the files to device buffers.
        At this moment, we do not instantiate tensors but just creating copies at device buffers with or without GDS.
        Users can instantiate and/or partition tensors with FilesBufferOnDevice returned by this function.
        The returned FilesBufferOnDevice owns the backing storage for tensors
        created from it. Clone/copy those tensors before FilesBufferOnDevice.close()
        if the tensor data must outlive the buffer.
        """
        self.framework.set_device(self.device)
        self._segment_visible_tensor_names = self._compute_segment_visible_tensor_names()
        keep_tensor = self._keep_tensor_name if self._has_tensor_visibility_filter() else None

        need_wait: List[LazyTensorFactory] = []
        factories: Dict[int, List[LazyTensorFactory]] = {}
        for i in range(0, self.pg.size()):
            factories[i] = []

        factory_idx_bits = math.ceil(math.log2(len(self.meta) + 1))
        lidx = 1
        for _, (meta, rank) in sorted(self.meta.items(), key=lambda x: x[0]):
            self_rank = self.pg.rank() == rank
            if self_rank:
                copier = self.copier_constructor(meta, self.device, self.framework)
                if keep_tensor is not None:
                    copier.set_tensor_filter(keep_tensor)
                byte_ranges = self._select_copy_byte_ranges(meta)
                if byte_ranges is not None:
                    copier.set_byte_ranges(byte_ranges)
            else:
                copier = None
            factory = LazyTensorFactory(
                meta,
                self.device,
                rank,
                self_rank,
                factory_idx_bits,
                lidx,
                copier,
                self.framework,
                disable_cache=self.disable_cache,
            )
            factory.submit_io(use_buf_register, max_copy_block_size)
            factories[rank].append(factory)
            if self_rank:
                need_wait.append(factory)
            lidx += 1
        for factory in need_wait:
            factory.wait_io(dtype=dtype, noalign=False)
        return FilesBufferOnDevice(
            factories,
            pg=self.pg,
            framework=self.framework,
            keep_tensor=keep_tensor,
        )


class MegaTensorsFileLoader(BaseMegaTensorsFileLoader):
    r"""Load .mega files lazily.

    Args:
        devcie (str): target device.
        pg (Optional[Any]): process group-like objects for distributed. None for single GPU use-cases.
        bbuf_size_kb (int): bounce buffer size for file copies.
        max_threads (int): maximum number of threads for memory copies.
        nogds (bool): if True, trun off GDS and fallback to pread with bounce buffer.
        debug_log (bool): enable debug logs.

    Examples:
        >> from megatensors.loader import MegaTensorsFileLoader
        >> src_files = download(target_dir, "gpt2")
        >> loader = MegaTensorsFileLoader(Device("cpu"), nogds=True, debug_log=True)
        >> loader.add_filenames({0: src_files})
        >> bufs = loader.copy_files_to_device()
        >> print(bufs.get_tensor(loader.get_keys()[0]))
        >> loader.close()
    """

    @classmethod
    def process_extension_config(
        cls, ext_config: Mapping[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """Map ``copier_type`` to ``nogds`` flag; pass rest through."""
        out = dict(ext_config)
        copier_type = out.pop("copier_type", "gds")
        out["nogds"] = copier_type != "gds"
        return out

    def __init__(
        self,
        pg: Optional[Any],
        device: str = "cpu",
        bbuf_size_kb: int = 16 * 1024,
        max_threads: int = 16,
        nogds: bool = False,
        set_numa: bool = True,
        disable_cache: bool = True,
        debug_log: bool = False,
        framework="pytorch",
        **kwargs,
    ):
        self.framework = get_framework_op(framework)
        self.pg = self.framework.get_process_group(pg)
        self.device = self.framework.get_device(device, self.pg)

        megacpp.set_debug_log(debug_log)

        if not nogds:
            if platform.system() == "Windows":
                copier_type = "dstorage"
            else:
                copier_type = "gds"
        elif self.device.type != DeviceType.CPU and is_unified_memory_system(
            self.framework
        ):
            # When GDS is unavailable, prefer the unified copier on systems
            # with shared CPU/GPU memory (e.g., DGX Spark) over the
            # bounce-buffer nogds path.
            copier_type = "unified"
        else:
            copier_type = "nogds"
        super().__init__(
            pg,
            self.device,
            copier_type,
            set_numa,
            disable_cache,
            framework,
            bbuf_size_kb=bbuf_size_kb,
            max_threads=max_threads,
            **kwargs,
        )


class mega_open:
    """
    Opens MEGA tensor artifact files lazily and returns tensors as requested.
    Tensors returned from this context are valid only while the context stays
    open. Clone/copy returned tensors before leaving the with block if the
    tensor data must be reused after __exit__ closes the backing buffer.

    Args:
        filenames (:obj:`str`|`list[str]`|`dict[int, str]`): The filename(s) or rank-file map to open
        framework (:obj:`str`): `pt`, `pytorch`, and `paddle` are only supported currently
        device (:obj:`str`, defaults to :obj:`"cpu"`): The device on which you want the tensors.
    """

    def __init__(
        self,
        filenames: Union[str, List[str], Dict[int, List[str]]],
        framework: str = "pt",
        pg: Optional[Any] = None,
        device: str = "cpu",
        nogds: bool = False,
        debug_log: bool = False,
        max_copy_block_size: int = 256 * 1024 * 1024,
    ):
        self.loader = MegaTensorsFileLoader(
            pg, device, nogds=nogds, debug_log=debug_log, framework=framework
        )
        file_dict: Dict[int, List[str]] = {}
        if isinstance(filenames, str):
            file_dict = {0: resolve_artifacts(filenames)}
        if isinstance(filenames, list):
            expanded: List[str] = []
            for filename in filenames:
                expanded.extend(resolve_artifacts(filename))
            file_dict = {0: expanded}
        elif isinstance(filenames, dict):
            file_dict = {
                rank: [
                    artifact
                    for filename in rank_filenames
                    for artifact in resolve_artifacts(filename)
                ]
                for rank, rank_filenames in filenames.items()
            }
        self.loader.add_filenames(file_dict)
        self.fb = self.loader.copy_files_to_device(
            max_copy_block_size=max_copy_block_size
        )

    def metadata(self) -> Dict[str, Dict[str, str]]:
        ret = {}
        for filename, (metadata, _) in self.loader.meta.items():
            ret[filename] = metadata.metadata
        return ret

    def keys(self) -> List[str]:
        return list(self.fb.key_to_rank_lidx.keys())

    def get_tensor_wrapped(self, name: str) -> TensorBase:
        """Return a wrapped tensor by name.

        Clone/copy the returned tensor before leaving the context manager if
        the tensor data must be used after the context closes.
        """
        return self.fb.get_tensor_wrapped(name)

    def get_tensor(self, name: str) -> Any:
        """Return a tensor by name.

        Clone/copy the returned tensor before leaving the context manager if
        the tensor data must be used after the context closes.
        """
        return self.get_tensor_wrapped(name).get_raw()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.fb:
            self.fb.close()
        if self.loader:
            self.loader.close()
