#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from megatensors import convert_model, load_state_dict, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hugging Face causal-LM generation with weights loaded from a MEGA artifact."
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
        help="Convert the HF safetensors directory to MEGA before generation",
    )
    parser.add_argument(
        "--mega-dir",
        type=Path,
        default=None,
        help="Output directory used with --convert. Defaults to <model_dir>/mega",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Target device for MEGA-loaded weights",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--nogds", action="store_true", help="Force NoGDS loading")
    return parser.parse_args()


def torch_dtype(name: str):
    return {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_mega_model(
    model_dir: Path,
    mega_path: Path,
    device: str,
    dtype,
    trust_remote_code: bool,
    nogds: bool,
):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    init_kwargs = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        init_kwargs["torch_dtype"] = dtype
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, **init_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_config(config, **init_kwargs)
    state = load_state_dict(
        str(mega_path),
        device=device,
        framework="pt",
        nogds=nogds,
        borrow=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing}, unexpected={unexpected}")
    setattr(model, "_mega_state_owner", state)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
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
    model = build_mega_model(
        model_dir,
        mega_path,
        args.device,
        dtype,
        args.trust_remote_code,
        args.nogds,
    )
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    generate_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.do_sample:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    generated = model.generate(**inputs, **generate_kwargs)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
