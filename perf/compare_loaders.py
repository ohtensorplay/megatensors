# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


WORKER = r"""
import argparse
import gc
import json
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["hf", "fastsafe", "mega"], required=True)
parser.add_argument("--path", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--nogds", action="store_true")
args = parser.parse_args()

if args.kind == "hf":
    from safetensors.torch import load_file
elif args.kind == "fastsafe":
    from fastsafetensors import fastsafe_open
else:
    from megatensors import mega_open

best = None
for _ in range(args.rounds):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    if args.kind == "hf":
        state = load_file(args.path, device=args.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ntensors = len(state)
        nbytes = sum(t.numel() * t.element_size() for t in state.values())
        del state
    else:
        opener = fastsafe_open if args.kind == "fastsafe" else mega_open
        with opener(args.path, framework="pt", device=args.device, nogds=args.nogds) as f:
            keys = list(f.keys())
            tensors = [f.get_tensor(k) for k in keys]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            ntensors = len(tensors)
            nbytes = sum(t.numel() * t.element_size() for t in tensors)
    if best is None or elapsed < best["seconds"]:
        best = {"seconds": elapsed, "tensors": ntensors, "bytes": nbytes}

best["kind"] = args.kind
best["gib_per_s"] = best["bytes"] / (1024 ** 3) / best["seconds"]
print(json.dumps(best, sort_keys=True))
"""


def run_one(kind: str, path: Path, device: str, rounds: int, nogds: bool) -> dict:
    cmd = [
        sys.executable,
        "-c",
        WORKER,
        "--kind",
        kind,
        "--path",
        str(path),
        "--device",
        device,
        "--rounds",
        str(rounds),
    ]
    if nogds:
        cmd.append("--nogds")
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HF safetensors, fastsafetensors, and MEGA load throughput."
    )
    parser.add_argument("--safetensors", required=True, type=Path)
    parser.add_argument("--mega", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--nogds", action="store_true")
    args = parser.parse_args()

    results = [
        run_one("hf", args.safetensors, args.device, args.rounds, args.nogds),
        run_one("fastsafe", args.safetensors, args.device, args.rounds, args.nogds),
        run_one("mega", args.mega, args.device, args.rounds, args.nogds),
    ]
    best = max(results, key=lambda item: item["gib_per_s"])
    print("kind       seconds    GiB/s    tensors    bytes")
    for item in results:
        marker = "*" if item is best else " "
        print(
            f"{marker}{item['kind']:<9} "
            f"{item['seconds']:>8.4f} "
            f"{item['gib_per_s']:>8.2f} "
            f"{item['tensors']:>8} "
            f"{item['bytes']:>12}"
        )


if __name__ == "__main__":
    main()
