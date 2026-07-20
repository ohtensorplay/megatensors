"""Public MEGA Hub API."""

from __future__ import annotations

from ._hub._snapshot_download import snapshot_download
from ._hub import (
    HFCacheInfo,
    HFSummaryWriter,
    HUGGINGFACE_CO_URL_HOME,
    HUGGINGFACE_CO_URL_TEMPLATE,
    HfApi,
    HfFileMetadata,
    HfFileSystem,
    HfFileSystemFile,
    HfFileSystemResolvedPath,
    HfFileSystemStreamFile,
    HfUri,
    attach_huggingface_oauth,
    check_cli_update,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    hf_raise_for_status,
    parse_hf_uri,
    parse_huggingface_oauth,
    repo_type_and_id_from_hf_id,
    typer_factory,
)
from ._hub.file_download import mega_hub_download, mega_hub_url
from ._hub.mega_api import MegaApi
from ._hub.mega_file_system import MegaFileSystem


__all__ = [
    "MegaApi",
    "MegaFileSystem",
    "mega_hub_download",
    "mega_hub_url",
    "snapshot_download",
    "HFCacheInfo",
    "HFSummaryWriter",
    "HUGGINGFACE_CO_URL_HOME",
    "HUGGINGFACE_CO_URL_TEMPLATE",
    "HfApi",
    "HfFileMetadata",
    "HfFileSystem",
    "HfFileSystemFile",
    "HfFileSystemResolvedPath",
    "HfFileSystemStreamFile",
    "HfUri",
    "attach_huggingface_oauth",
    "check_cli_update",
    "get_hf_file_metadata",
    "hf_hub_download",
    "hf_hub_url",
    "hf_raise_for_status",
    "parse_hf_uri",
    "parse_huggingface_oauth",
    "repo_type_and_id_from_hf_id",
    "typer_factory",
]
