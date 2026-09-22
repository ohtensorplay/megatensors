from pathlib import Path

import pytest

tp = pytest.importorskip("tensorplay")

from megatensors import load_state_dict, mega_open, write_tensorplay_file
from megatensors.frameworks import get_framework_op


def test_tensorplay_framework_aliases():
    assert get_framework_op("tp").get_name() == "tensorplay"
    assert get_framework_op("tensorplay").get_name() == "tensorplay"


def test_tensorplay_mega_round_trip(tmp_path: Path):
    values = {
        "float": tp.tensor([[1.0, 2.0]], dtype=tp.float32),
        "bf16": tp.tensor([3.0, 4.0], dtype=tp.bfloat16),
        "integer": tp.tensor([5, 6], dtype=tp.int64),
    }
    filename = tmp_path / "weights.mega"

    write_tensorplay_file(filename, values)

    state = load_state_dict(
        str(filename), framework="tensorplay", device="cpu", nogds=True
    )
    assert list(state) == list(values)
    for name, value in values.items():
        assert state[name].dtype == value.dtype
        assert tp.allclose(state[name], value)

    with mega_open(str(filename), framework="tp", device="cpu", nogds=True) as artifact:
        assert artifact.keys() == list(values)
