# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID

from .api import append_footer_overlay
from .common import (
    MEGA_TRUST_CERTIFICATE_FORMAT,
    MEGA_TRUST_SIGNATURE_STATEMENT_PREFIX,
    MegaTensorsMetadata,
)
from .convert import resolve_artifacts
from .frameworks import get_framework_op


@dataclass(frozen=True)
class SigningConfig:
    private_key_pem: str | bytes
    leaf_certificate_pem: str | bytes
    publisher: str
    model_id: str
    chain_pem: str | bytes = ""
    algorithm: str = "sha256-rsa-pss"
    created_at: int | None = None
    expires_at: int | None = None
    key_password: str | bytes | None = None
    generation: int = 1

    @classmethod
    def from_pem_bundle(
        cls,
        bundle_pem: str | bytes,
        *,
        model_id: str,
        key_password: str | bytes | None = None,
        algorithm: str = "sha256-rsa-pss",
        expires_at: int | None = None,
        generation: int = 1,
    ) -> "SigningConfig":
        bundle = _as_bytes(bundle_pem)
        certs = x509.load_pem_x509_certificates(bundle)
        if not certs:
            raise ValueError("signing bundle must contain a leaf certificate")
        leaf = certs[0]
        chain = b"".join(cert.public_bytes(serialization.Encoding.PEM) for cert in certs[1:])
        password = key_password
        if password is None:
            password = os.environ.get("MEGA_SIGNING_PASSWORD")
        return cls(
            private_key_pem=bundle,
            leaf_certificate_pem=leaf.public_bytes(serialization.Encoding.PEM),
            chain_pem=chain,
            publisher=_certificate_publisher(leaf),
            model_id=model_id,
            algorithm=algorithm,
            expires_at=expires_at,
            key_password=password,
            generation=generation,
        )


def sign_artifact(filename: str | Path, config: SigningConfig) -> list[Path]:
    """Sign one MEGA shard or every shard referenced by a MEGA JSON index."""
    paths = [Path(p) for p in resolve_artifacts(filename)]
    for path in paths:
        _sign_one(path, config)
    return paths


def _sign_one(path: Path, config: SigningConfig) -> None:
    leaf_pem = _as_bytes(config.leaf_certificate_pem)
    chain_pem = _as_text(config.chain_pem)
    private_key = serialization.load_pem_private_key(
        _as_bytes(config.private_key_pem),
        password=_password_bytes(config.key_password),
    )
    leaf_cert = x509.load_pem_x509_certificate(leaf_pem)
    _validate_signing_certificate(leaf_cert, private_key)

    now = int(time.time()) if config.created_at is None else int(config.created_at)
    expires_at = (
        now + 365 * 24 * 60 * 60
        if config.expires_at is None
        else int(config.expires_at)
    )
    if expires_at <= now:
        raise ValueError("expires_at must be greater than created_at")

    fw = get_framework_op("pt")
    metadata = MegaTensorsMetadata.from_file(str(path), fw)
    payload_sha256 = metadata.compute_payload_sha256()
    statement = _statement(
        publisher=config.publisher,
        model_id=config.model_id,
        payload_sha256=payload_sha256,
        created_at=now,
        expires_at=expires_at,
    )
    signature = _sign_statement(private_key, statement.encode("utf-8"), config.algorithm)

    append_footer_overlay(
        str(path),
        {
            "mega.hash.payload.sha256": payload_sha256,
            "mega.trust.certificate.format": MEGA_TRUST_CERTIFICATE_FORMAT,
            "mega.trust.certificate.leaf_pem": leaf_pem.decode("utf-8"),
            "mega.trust.certificate.chain_pem": chain_pem,
            "mega.trust.signature.statement": statement,
            "mega.trust.signature.algorithm": config.algorithm,
            "mega.trust.signature.value": base64.b64encode(signature).decode("ascii"),
        },
        generation=int(config.generation),
    )


def _statement(
    *,
    publisher: str,
    model_id: str,
    payload_sha256: str,
    created_at: int,
    expires_at: int,
) -> str:
    for name, value in {
        "publisher": publisher,
        "model_id": model_id,
        "payload_sha256": payload_sha256,
    }.items():
        if not value or "\n" in value or "\r" in value:
            raise ValueError(f"{name} must be non-empty and single-line")
    return (
        MEGA_TRUST_SIGNATURE_STATEMENT_PREFIX +
        "format_version=1\n"
        "artifact_kind=model\n"
        f"publisher={publisher}\n"
        f"model_id={model_id}\n"
        f"payload_sha256={payload_sha256.lower()}\n"
        f"created_at={int(created_at)}\n"
        f"expires_at={int(expires_at)}\n"
    )


def _sign_statement(private_key, statement: bytes, algorithm: str) -> bytes:
    if algorithm == "sha256-rsa-pss":
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("sha256-rsa-pss requires an RSA private key")
        return private_key.sign(
            statement,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
    if algorithm in {"sha256-rsa-pkcs1", "sha256-rsa"}:
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError(f"{algorithm} requires an RSA private key")
        return private_key.sign(statement, padding.PKCS1v15(), hashes.SHA256())
    if algorithm == "sha256-ecdsa":
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise ValueError("sha256-ecdsa requires an EC private key")
        return private_key.sign(statement, ec.ECDSA(hashes.SHA256()))
    raise ValueError(f"unsupported signature algorithm: {algorithm}")


def _validate_signing_certificate(cert: x509.Certificate, private_key) -> None:
    cert_pub = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_pub = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if cert_pub != key_pub:
        raise ValueError("leaf certificate public key does not match private key")
    try:
        ku = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound as exc:
        raise ValueError("leaf certificate must include keyUsage") from exc
    if not ku.digital_signature:
        raise ValueError("leaf certificate keyUsage must allow digitalSignature")
    try:
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound as exc:
        raise ValueError("leaf certificate must include extendedKeyUsage") from exc
    if ExtendedKeyUsageOID.CODE_SIGNING not in eku:
        raise ValueError("leaf certificate extendedKeyUsage must include codeSigning")


def _certificate_publisher(cert: x509.Certificate) -> str:
    publisher = cert.subject.rfc4514_string()
    if not publisher:
        raise ValueError("leaf certificate subject must not be empty")
    return publisher


def _as_bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _as_text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _password_bytes(value: str | bytes | None) -> bytes | None:
    if value is None:
        return None
    return value if isinstance(value, bytes) else value.encode("utf-8")
