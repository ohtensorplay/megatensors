# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
from typing import Any, Dict

import pytest

# Import fixtures from parent conftest so they are available in this directory
from conftest import dev_init, input_files  # noqa: F401
from threefs.mock_reader import MockFileReader
from threefs.mock_reader import extract_mount_point as mock_extract_mount_point

from megatensors import cpp as megacpp
from megatensors.common import SingleGroup, is_gpu_found, resolve_runtime_lib_name
from megatensors.cpp import load_library_functions
from megatensors.frameworks import FrameworkOpBase, get_framework_op
from megatensors.st_types import Device


@pytest.fixture(autouse=True, scope="session")
def mock_3fs_reader():
    """If fastsafetensor_3fs_reader is not installed, inject a mock module."""
    try:
        import fastsafetensor_3fs_reader  # noqa: F401
    except ImportError:
        mock_module = types.ModuleType("fastsafetensor_3fs_reader")
        mock_module.ThreeFSFileReader = MockFileReader
        mock_module.MockFileReader = MockFileReader
        mock_module.extract_mount_point = mock_extract_mount_point
        sys.modules["fastsafetensor_3fs_reader"] = mock_module
    yield


load_library_functions(resolve_runtime_lib_name())
FRAMEWORK = get_framework_op(os.getenv("TEST_MEGATENSORS_FRAMEWORK", "please set"))


def get_device(framework: FrameworkOpBase):
    return Device.from_str("cpu"), False


def load_megatensors_file(
    filename: str,
    device: Device,
    framework: FrameworkOpBase,
) -> Dict[str, Any]:
    from megatensors import load_state_dict

    return load_state_dict(filename, device=device.as_str(), nogds=True)


def tensors_equal(actual: Any, expected: Any, framework: FrameworkOpBase) -> bool:
    """Compare raw tensors (torch.Tensor / paddle.Tensor) for equality."""
    if framework.get_name() == "pytorch":
        import torch

        return bool(torch.all(actual.eq(expected)))
    elif framework.get_name() == "paddle":
        import paddle

        return bool(paddle.all(actual == expected))
    else:
        raise Exception(f"unknown framework: {framework.get_name()}")


@pytest.fixture(scope="session")
def framework() -> FrameworkOpBase:
    return FRAMEWORK


@pytest.fixture(scope="function")
def megacpp_log() -> None:
    megacpp.set_debug_log(True)
