#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from megatensors import convert_model, load_state_dict, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that MEGA weights reproduce Hugging Face causal-LM inference."
    )
    parser.add_argument("model_dir", type=Path, help="Hugging Face model directory")
    parser.add_argument(
        "--mega",
        type=Path,
        default=None,
        help="MEGA shard or .mega.index.json. Defaults to <model_dir>/mega/model.mega.index.json",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert the HF safetensors directory to MEGA before verification",
    )
    parser.add_argument(
        "--mega-dir",
        type=Path,
        default=None,
        help="Output directory used with --convert. Defaults to <model_dir>/mega",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--generate", action="store_true", help="Also print a short MEGA generation")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--nogds", action="store_true", help="Force NoGDS loading")
    return parser.parse_args()


def torch_dtype(name: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.inference_mode()
def logits_for(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**inputs).logits[:, -1, :].detach().float().cpu()


def build_mega_model(model_dir: Path, mega_path: Path, device: str, dtype, trust_remote_code: bool, nogds: bool):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=dtype if dtype != "auto" else None,
                trust_remote_code=trust_remote_code,
            )
    except Exception:
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=dtype if dtype != "auto" else None,
            trust_remote_code=trust_remote_code,
        )
    state = load_state_dict(str(mega_path), device=device, framework="pt", nogds=nogds, borrow=True)
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing}, unexpected={unexpected}")
    setattr(model, "_mega_state_owner", state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    mega_dir = args.mega_dir.resolve() if args.mega_dir else model_dir / "mega"
    mega_path = args.mega.resolve() if args.mega else mega_dir / "model.mega.index.json"
    dtype = torch_dtype(args.dtype)

    if args.convert:
        result = convert_model(model_dir, mega_dir, compression="none")
        mega_path = result.index_path

    if not mega_path.exists():
        raise FileNotFoundError(f"{mega_path} does not exist; pass --convert or --mega")

    tokenizer = load_tokenizer(str(mega_path))
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    hf_model.eval()
    hf_logits = logits_for(hf_model, inputs)
    del hf_model
    clear_cuda()

    mega_model = build_mega_model(
        model_dir,
        mega_path,
        args.device,
        dtype,
        args.trust_remote_code,
        args.nogds,
    )
    mega_logits = logits_for(mega_model, inputs)
    max_abs = (hf_logits - mega_logits).abs().max().item()
    allclose = torch.allclose(hf_logits, mega_logits, rtol=args.rtol, atol=args.atol)
    print(f"hf_logits_shape={tuple(hf_logits.shape)}")
    print(f"mega_logits_shape={tuple(mega_logits.shape)}")
    print(f"max_abs_diff={max_abs:.6g}")
    print(f"allclose={allclose} rtol={args.rtol} atol={args.atol}")
    if not allclose:
        raise SystemExit(1)

    if args.generate:
        generated = mega_model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
