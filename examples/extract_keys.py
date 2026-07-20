#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys

from megatensors import mega_open


def _iter_mega_files(path: str):
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.endswith(".mega"):
                    yield os.path.join(root, filename)
    elif os.path.exists(path) and path.endswith(".mega"):
        yield path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("specify a .mega file or directory containing .mega files")
        sys.exit(1)
    for filename in _iter_mega_files(sys.argv[1]):
        with mega_open(filename, device="cpu", nogds=True) as artifact:
            for key in artifact.keys():
                tensor = artifact.get_tensor(key)
                print(f'"{filename}","{key}",{tuple(tensor.shape)},{tensor.dtype}')
