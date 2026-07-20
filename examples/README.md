# Examples

These examples are the supported entry points in this directory:

- `causal_lm_generate.py`: build a Hugging Face causal-LM skeleton, load weights from MEGA, load the tokenizer from MEGA, and run generation.
- `hf_inference_verify.py`: compare Hugging Face logits against the same model loaded from MEGA.
- `sign_and_verify.py`: sign a MEGA artifact with a PEM bundle and verify it against external trust roots.
- `extract_keys.py`: print tensor names, shapes, and dtypes from `.mega` files.

These files are source-tree examples and are not included in the installed wheel. All commands below assume you are running from the repository root and `python` resolves to the environment where MEGA is installed.

## Setup

```bash
python -m pip install -e "./megatensors[test]"
```

## Generate From MEGA

Convert a Hugging Face safetensors directory to MEGA and run GPU generation:

```bash
python megatensors/examples/causal_lm_generate.py \
  Qwen3.5-0.8B \
  --convert \
  --mega-dir Qwen3.5-0.8B/mega \
  --device cuda:0 \
  --trust-remote-code \
  --prompt "Write a short summary of MEGA format." \
  --max-new-tokens 64
```

Run generation from an existing artifact:

```bash
python megatensors/examples/causal_lm_generate.py \
  Qwen3.5-0.8B \
  --mega Qwen3.5-0.8B/mega/model.mega.index.json \
  --device cuda:0 \
  --trust-remote-code \
  --prompt "Hello, my name is"
```

Notes:

- The tokenizer is loaded from the MEGA artifact with `load_tokenizer(...)`.
- The model structure still comes from the Hugging Face `config.json` and class implementation.
- `--nogds` forces the NoGDS backend when you do not want to use `cufile`.

## Verify Hugging Face Equivalence

This script loads the Hugging Face model and the MEGA-loaded model, compares final-token logits, and can also print a short generation:

```bash
python megatensors/examples/hf_inference_verify.py \
  Qwen3.5-0.8B \
  --mega Qwen3.5-0.8B/mega/model.mega.index.json \
  --device cuda:0 \
  --trust-remote-code \
  --generate
```

## Sign And Verify

Sign a MEGA artifact with a PEM bundle that contains:

- the private key
- the leaf code-signing certificate
- the optional intermediate chain

Then verify the signed shards with external trust roots:

```bash
python megatensors/examples/sign_and_verify.py \
  Qwen3.5-0.8B/mega/model.mega.index.json \
  --bundle /path/to/official_signing_bundle.pem \
  --trusted-roots /path/to/official_root_ca.pem \
  --model-id qwen3.5-0.8b
```

If the private key is encrypted, provide the password with `MEGA_SIGNING_PASSWORD` or `--key-password`.

Important:

- Trust roots are external policy. They must not come from the artifact itself.
- A signature from an unknown issuer still verifies cryptographically as a signature, but it remains a source-risk artifact.
- Verification prints one line per shard with `trusted=...`, `risk=...`, `publisher=...`, and `payload_sha256=...`.
