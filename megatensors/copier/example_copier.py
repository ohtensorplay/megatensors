# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from .. import cpp as megacpp
from ..common import MegaTensorsMetadata
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface, DummyDeviceBuffer


class ExampleCopier(CopierInterface):
    def __init__(
        self,
        metadata: MegaTensorsMetadata,
        device: Device,
        reader,
        framework: FrameworkOpBase,
    ):
        self.metadata = metadata

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> megacpp.gds_device_buffer:
        return DummyDeviceBuffer()

    def wait_io(
        self,
        gbuf: megacpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        # get tensor
        res: Dict[str, TensorBase] = {}
        return res


def new_gds_file_copier(
    device: Device,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    nogds: bool = False,
):
    # reader = example_reader()
    reader: Any = {}

    def construct_copier(
        metadata: MegaTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return ExampleCopier(metadata, device, reader, framework)

    return construct_copier
