# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated

from megatensors.convert import convert_model
from megatensors.signing import SigningConfig, sign_artifact

from ._framework import Argument, Option
from ._output import out


def convert(
    model_dir: Annotated[Path, Argument(help="Model directory to convert.")],
    output_dir: Annotated[Path | None, Option("--output-dir", "-o")] = None,
    basename: Annotated[str, Option("--basename")] = "model",
    max_shard_size: Annotated[str, Option("--max-shard-size")] = "5GB",
    alignment: Annotated[int, Option("--alignment")] = 4096,
    shard_metadata: Annotated[str, Option("--shard-metadata")] = "first",
    compression: Annotated[str, Option("--compression")] = "none",
    compression_level: Annotated[int, Option("--compression-level")] = 3,
    compression_min_ratio: Annotated[float, Option("--compression-min-ratio")] = 0.98,
    sign_bundle: Annotated[
        Path | None,
        Option("--sign-bundle", help="PEM bundle: private key, leaf certificate, optional chain."),
    ] = None,
) -> None:
    """Convert a model folder to MEGA artifacts."""
    signing = _signing_config(sign_bundle, _default_model_id(model_dir))
    result = convert_model(
        model_dir,
        output_dir,
        basename=basename,
        max_shard_size=max_shard_size,
        alignment=alignment,
        shard_metadata=shard_metadata,
        compression=compression,
        compression_level=compression_level,
        compression_min_ratio=compression_min_ratio,
        signing=signing,
    )
    gib = result.payload_nbytes / (1024**3)
    rate = gib / result.elapsed_seconds if result.elapsed_seconds > 0 else 0.0
    out.result(
        "Model converted",
        tensors=result.tensor_count,
        shards=len(result.shard_paths),
        payload_gib=round(gib, 3),
        seconds=round(result.elapsed_seconds, 2),
        gib_per_second=round(rate, 2),
        index=result.index_path,
    )


def sign(
    artifact: Annotated[Path, Argument(help="MEGA shard or .mega.index.json.")],
    bundle: Annotated[Path, Option("--bundle", help="PEM bundle: private key, leaf certificate, optional chain.")],
) -> None:
    """Sign a MEGA artifact."""
    signing = _signing_config(bundle, _default_model_id(artifact))
    assert signing is not None
    paths = sign_artifact(artifact, signing)
    out.result("Artifact signed", artifact=artifact, shards=len(paths))


def _signing_config(bundle: Path | None, model_id: str) -> SigningConfig | None:
    if bundle is None:
        return None
    return SigningConfig.from_pem_bundle(bundle.read_bytes(), model_id=model_id)


def _default_model_id(path: Path) -> str:
    name = path.name
    if name.endswith(".mega.index.json"):
        return name[: -len(".mega.index.json")]
    if name.endswith(".mega"):
        return name[: -len(".mega")]
    return name
