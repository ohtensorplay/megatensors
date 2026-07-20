# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import pytest

from megatensors import cpp as megacpp
from megatensors.common import SingleGroup, is_gpu_found, resolve_runtime_lib_name
from megatensors.cpp import load_library_functions
from megatensors.frameworks import FrameworkOpBase, get_framework_op
from megatensors.st_types import Device

# Add tests directory to path to import platform_utils
TESTS_DIR = os.path.dirname(__file__)
from platform_utils import get_platform_info

REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
DATA_DIR = os.path.join(REPO_ROOT, ".testdata")
TF_DIR = os.path.join(DATA_DIR, "transformers_cache")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
os.makedirs(TF_DIR, 0o777, True)
os.makedirs(TMP_DIR, 0o777, True)
os.makedirs(GENERATED_DIR, 0o777, True)

load_library_functions(resolve_runtime_lib_name())
FRAMEWORK = get_framework_op(os.getenv("TEST_MEGATENSORS_FRAMEWORK", "please set"))

# Print platform information at test startup
platform_info = get_platform_info()
print("\n" + "=" * 60)
print("Platform Detection:")
print("=" * 60)
for key, value in platform_info.items():
    print(f"  {key}: {value}")
print("=" * 60 + "\n")


@pytest.fixture(scope="session", autouse=True)
def framework() -> FrameworkOpBase:
    return FRAMEWORK


@pytest.fixture(scope="session")
def input_files() -> List[str]:
    if os.environ.get("MEGATENSORS_TEST_USE_HF_GPT2") != "1":
        return [_ensure_tiny_gpt2_megatensors(FRAMEWORK)]

    gpt_dir = os.path.join(TF_DIR, "models--gpt2")
    if not os.path.exists(gpt_dir):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        AutoModelForCausalLM.from_pretrained(
            "gpt2", trust_remote_code=True, use_megatensors=True, cache_dir=TF_DIR
        )
        AutoTokenizer.from_pretrained("gpt2", cache_dir=TF_DIR)
    src_files = []
    for dir, _, files in os.walk(gpt_dir):
        for filename in files:
            if filename.endswith(".megatensors"):
                src_files.append(f"{dir}/{filename}")
                print(src_files[-1])
    return src_files


def _ensure_tiny_gpt2_megatensors(framework: FrameworkOpBase) -> str:
    filename = os.path.join(
        GENERATED_DIR, f"tiny-gpt2-{framework.get_name()}.mega"
    )
    if os.path.exists(filename):
        return filename

    tmp_filename = f"{filename}.{os.getpid()}.tmp"
    if framework.get_name() == "pytorch":
        import torch

        dtype = torch.float16

        def make_tensor(rows: int, cols: int, offset: int = 0):
            return torch.arange(rows * cols, dtype=dtype).reshape(rows, cols) + offset

        def make_bias(size: int, offset: int = 0):
            return torch.arange(size, dtype=dtype) + offset

    elif framework.get_name() == "paddle":
        pytest.skip("MEGA test fixture writer currently creates PyTorch payloads")

    else:
        raise Exception(f"Unknown framework: {framework.get_name()}")

    from test_mega_format import _write_mega

    tensors = []
    for layer in range(12):
        base = layer * 1000
        for name, tensor in (
            (f"h.{layer}.mlp.c_proj.weight", make_tensor(8, 16, base)),
            (f"h.{layer}.mlp.c_fc.weight", make_tensor(16, 8, base + 100)),
            (f"h.{layer}.attn.c_proj.weight", make_tensor(8, 8, base + 200)),
            (f"h.{layer}.attn.c_proj.bias", make_bias(8, base + 300)),
        ):
            tensors.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "logical_dtype": "F16",
                    "storage_format": "raw_dense",
                    "logical_nbytes": tensor.numel() * tensor.element_size(),
                    "data": tensor.contiguous().numpy().tobytes(),
                }
            )

    _write_mega(tmp_filename, tensors, metadata={"general.name": "tiny-gpt2"})
    os.replace(tmp_filename, filename)
    return filename


@pytest.fixture(scope="session", autouse=True)
def pg():
    rank = int(os.environ.get("RANK", "0"))
    if is_gpu_found():
        dev_str = f"cuda:{rank}" if FRAMEWORK.get_name() == "pytorch" else f"gpu:{rank}"
    else:
        dev_str = "cpu"
    FRAMEWORK.set_device(Device.from_str(dev_str))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    backend = "nccl" if is_gpu_found() else "gloo"
    if world_size > 1:
        if FRAMEWORK.get_name() == "pytorch":
            import torch.distributed as dist

            dist.init_process_group(backend=backend)
            dist.barrier()
            return dist.group.WORLD
        elif FRAMEWORK.get_name() == "paddle":
            # The following code can only be successfully
            # executed by running the code using
            # `python -m paddle.distributed.launch`
            import paddle.distributed as dist

            dist.init_parallel_env()
            return dist.new_group(ranks=list(range(world_size)), backend=backend)
    return SingleGroup()


@pytest.fixture(scope="session", autouse=True)
def dev_init() -> None:
    rank = int(os.environ.get("RANK", "0"))
    if is_gpu_found():
        dev_str = f"cuda:{rank}" if FRAMEWORK.get_name() == "pytorch" else f"gpu:{rank}"
    else:
        dev_str = "cpu"
    FRAMEWORK.set_device(Device.from_str(dev_str))


@pytest.fixture(scope="function")
def megacpp_log() -> None:
    megacpp.set_debug_log(True)


@pytest.fixture(scope="function")
def tmp_dir() -> str:
    return TMP_DIR
