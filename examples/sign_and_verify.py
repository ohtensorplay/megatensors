#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from pathlib import Path

from megatensors import MegaTensorsMetadata, SigningConfig, sign_artifact
from megatensors.frameworks import get_framework_op


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign a MEGA artifact with a PEM bundle and optionally verify it against trusted roots."
    )
    parser.add_argument("artifact", type=Path, help="MEGA shard or .mega.index.json")
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="PEM bundle containing private key, leaf certificate, and optional chain",
    )
    parser.add_argument(
        "--trusted-roots",
        type=Path,
        default=None,
        help="PEM trust roots used for verification",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Expected model_id in the signed statement. Defaults to artifact basename",
    )
    parser.add_argument(
        "--algorithm",
        default="sha256-rsa-pss",
        choices=["sha256-rsa-pss", "sha256-rsa-pkcs1", "sha256-rsa", "sha256-ecdsa"],
    )
    parser.add_argument(
        "--expires-days",
        type=int,
        default=365,
        help="Statement validity window in days",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=1,
        help="Footer overlay generation written by append_footer_overlay",
    )
    parser.add_argument(
        "--key-password",
        default=None,
        help="Private key password. If omitted, MEGA_SIGNING_PASSWORD is used.",
    )
    parser.add_argument(
        "--strict-verify",
        action="store_true",
        help="Raise on verification failure instead of returning a warning result",
    )
    return parser.parse_args()


def _default_model_id(path: Path) -> str:
    name = path.name
    if name.endswith(".mega.index.json"):
        return name[: -len(".mega.index.json")]
    if name.endswith(".mega"):
        return name[: -len(".mega")]
    return path.stem


def _verify(paths: list[str], trusted_roots: Path, model_id: str, strict: bool) -> int:
    fw = get_framework_op("pt")
    trusted_roots_pem = trusted_roots.read_text(encoding="utf-8")
    failures = 0
    for path in paths:
        metadata = MegaTensorsMetadata.from_file(path, fw)
        result = metadata.verify_trust(
            trusted_roots_pem,
            strict=strict,
            warn=not strict,
            allowed_model_ids=[model_id],
        )
        statement = result.get("statement", {})
        print(
            f"{path}: "
            f"trusted={bool(result.get('trusted'))} "
            f"risk={result.get('risk')} "
            f"publisher={statement.get('publisher', '')} "
            f"model_id={statement.get('model_id', '')} "
            f"payload_sha256={result.get('payload_sha256', '')}"
        )
        if not bool(result.get("trusted")):
            failures += 1
    return failures


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    model_id = args.model_id or _default_model_id(artifact)
    expires_at = int(time.time()) + max(int(args.expires_days), 1) * 24 * 60 * 60
    config = SigningConfig.from_pem_bundle(
        args.bundle.read_bytes(),
        model_id=model_id,
        key_password=args.key_password,
        algorithm=args.algorithm,
        expires_at=expires_at,
        generation=args.generation,
    )
    signed_paths = sign_artifact(artifact, config)
    print(f"signed {len(signed_paths)} shard(s)")
    for path in signed_paths:
        print(path)

    if args.trusted_roots is None:
        return

    failures = _verify([str(path) for path in signed_paths], args.trusted_roots, model_id, args.strict_verify)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
