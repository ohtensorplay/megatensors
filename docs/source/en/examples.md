# Examples


Example scripts live under `megatensors/examples` in the MEGA source tree. They are not installed with the Python wheel, so run this page from a source checkout after installing the package and example dependencies:

```bash
python -m pip install -e "./megatensors[test]"
```

The commands below assume the current directory is the repository root and `python` resolves to that environment.

## Generate text from a MEGA artifact

```bash
python megatensors/examples/causal_lm_generate.py \
  Qwen3.5-0.8B \
  --mega Qwen3.5-0.8B/mega/model.mega.index.json \
  --device cuda:0 \
  --trust-remote-code \
  --prompt "Hello, my name is"
```

Use `--nogds` if the host does not have GPUDirect Storage configured:

```bash
python megatensors/examples/causal_lm_generate.py \
  Qwen3.5-0.8B \
  --mega Qwen3.5-0.8B/mega/model.mega.index.json \
  --device cuda:0 \
  --nogds \
  --prompt "Explain tensor loading in one sentence."
```

## Verify Hugging Face equivalence

Run a short comparison against the source Hugging Face directory:

```bash
python megatensors/examples/hf_inference_verify.py \
  Qwen3.5-0.8B \
  --mega Qwen3.5-0.8B/mega/model.mega.index.json \
  --device cuda:0 \
  --trust-remote-code
```

Use this before publishing a converted artifact.

## Inspect tensor keys

```bash
python megatensors/examples/extract_keys.py Qwen3.5-0.8B/mega
```

This is useful when `load_model` fails because a model class expects different key names.

## Sign and verify

```bash
python megatensors/examples/sign_and_verify.py \
  Qwen3.5-0.8B/mega/model.mega.index.json \
  --bundle ./signing/release-bundle.pem \
  --trusted-roots ./signing/root-ca.pem \
  --model-id qwen3.5-0.8b
```

## Minimal Python smoke test

```python
from megatensors import load_state_dict, load_tokenizer

path = "Qwen3.5-0.8B/mega/model.mega.index.json"
state = load_state_dict(path, device="cpu")
tokenizer = load_tokenizer(path)

print(len(state))
print(tokenizer("hello").input_ids[:8])
```
