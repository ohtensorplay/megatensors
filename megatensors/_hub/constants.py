import os
import re
import typing
from typing import Literal
from urllib.parse import urlsplit


# Possible values for env variables


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def _is_true(value: str | None) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


# Constants for file downloads

PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
CONFIG_NAME = "config.json"
REPOCARD_NAME = "README.md"
EVAL_RESULTS_FOLDER = ".eval_results"
DEFAULT_ETAG_TIMEOUT = 10
DEFAULT_DOWNLOAD_TIMEOUT = 10
DEFAULT_REQUEST_TIMEOUT = 10
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024
MAX_HTTP_DOWNLOAD_SIZE = 50 * 1000 * 1000 * 1000  # 50 GB

# Constants for serialization

PYTORCH_WEIGHTS_FILE_PATTERN = "pytorch_model{suffix}.bin"  # Unsafe pickle: use safetensors instead
SAFETENSORS_WEIGHTS_FILE_PATTERN = "model{suffix}.safetensors"
TF2_WEIGHTS_FILE_PATTERN = "tf_model{suffix}.h5"

# Constants for safetensors repos

SAFETENSORS_SINGLE_FILE = "model.safetensors"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
SAFETENSORS_MAX_HEADER_LENGTH = 25_000_000

# Timeout of acquiring file lock and logging the attempt
FILELOCK_LOG_EVERY_SECONDS = 10

# Git-related constants

DEFAULT_REVISION = "main"
REGEX_COMMIT_OID = re.compile(r"[A-Fa-f0-9]{5,40}")

MEGA_URL_HOME = "https://mega.tensorplay.cn/"

_MEGA_DEFAULT_ENDPOINT = "https://mega.tensorplay.cn"
ENDPOINT = os.getenv("MEGA_ENDPOINT", _MEGA_DEFAULT_ENDPOINT).rstrip("/")
# The Worker exposes the immutable artifact endpoint behind the MEGA
# download/cache machinery. ``revision`` is intentionally a query parameter.
MEGA_URL_TEMPLATE = ENDPOINT + "/api/repos/{repo_id}/resolve/{filename}?revision={revision}"

# Source-compatibility spellings from ``huggingface_hub``. They deliberately
# resolve to MEGA's control plane: importing a compatibility name must not send
# repository traffic to huggingface.co.
HUGGINGFACE_CO_URL_HOME = MEGA_URL_HOME
HUGGINGFACE_CO_URL_TEMPLATE = MEGA_URL_TEMPLATE

# Hosts whose web URLs can be parsed into a ``mega://`` URI.
# Includes the public Hub host, the staging host, and the host of the
# currently configured ``ENDPOINT`` so that self-hosted / staging endpoints work too.
MEGA_URL_HOSTS: frozenset[str] = frozenset(
    host.lower() for host in (urlsplit(_MEGA_DEFAULT_ENDPOINT).hostname, urlsplit(ENDPOINT).hostname) if host
)

DATASETS_SERVER_ENDPOINT = os.environ.get("MEGA_DATASETS_SERVER_ENDPOINT", ENDPOINT + "/api/datasets")

MEGA_HEADER_X_REPO_COMMIT = "X-Repo-Commit"
MEGA_HEADER_X_LINKED_ETAG = "X-Linked-Etag"
MEGA_HEADER_X_LINKED_SIZE = "X-Linked-Size"
MEGA_HEADER_X_BILL_TO = "X-Mega-Bill-To"
MEGA_HEADER_X_INFERENCE_BILLING = "X-Mega-Inference-Billing"
MEGA_HEADER_X_INFERENCE_SESSION = "X-Mega-Session-Id"

INFERENCE_ENDPOINT = os.environ.get("MEGA_INFERENCE_ENDPOINT", ENDPOINT + "/api/inference")

# Public OpenAI-compatible routed inference data plane. Keep this separate from
# ``ENDPOINT``: mega.tensorplay.cn is the Hub control plane, while
# inference.tensorplay.cn never exposes Hub account or sub2api administration APIs.
INFERENCE_ROUTER_ENDPOINT = os.environ.get(
    "MEGA_INFERENCE_ROUTER_ENDPOINT", "https://inference.tensorplay.cn"
).rstrip("/")
INFERENCE_MODELS_ENDPOINT = os.environ.get(
    "MEGA_INFERENCE_MODELS_ENDPOINT", ENDPOINT + "/api/inference/models"
)

# See https://mega.tensorplay.cn/docs/inference-endpoints/index
INFERENCE_ENDPOINTS_ENDPOINT = os.environ.get(
    "MEGA_INFERENCE_ENDPOINTS_ENDPOINT", ENDPOINT + "/api/inference-endpoints"
)
INFERENCE_CATALOG_ENDPOINT = os.environ.get(
    "MEGA_INFERENCE_CATALOG_ENDPOINT", ENDPOINT + "/api/inference-endpoints/catalog"
)

INFERENCE_ENDPOINT_IMAGE_KEYS = [
    "custom",
    "mega",
    "megaNeuron",
    "llamacpp",
    "tei",
    "tgi",
    "tgiNeuron",
]

# Backwards-compatible name used by provider helpers. Routed OpenAI requests no
# longer encode the provider in the URL; the Router reads the HF-compatible
# ``owner/model:provider`` suffix instead.
INFERENCE_PROXY_TEMPLATE = INFERENCE_ROUTER_ENDPOINT

REPO_ID_SEPARATOR = "--"
# This separator is reserved for MEGA's on-disk repository identifiers.


REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPE_KERNEL = "kernel"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]
REPO_TYPES_WITH_KERNEL = REPO_TYPES + [REPO_TYPE_KERNEL]
SPACES_SDK_TYPES = ["gradio", "streamlit", "docker", "static"]

REPO_TYPES_URL_PREFIXES = {
    REPO_TYPE_DATASET: "datasets/",
    REPO_TYPE_SPACE: "spaces/",
    REPO_TYPE_KERNEL: "kernels/",
}
REPO_TYPES_MAPPING = {
    "datasets": REPO_TYPE_DATASET,
    "spaces": REPO_TYPE_SPACE,
    "models": REPO_TYPE_MODEL,
    "kernels": REPO_TYPE_KERNEL,
}

# MEGA Hub URI protocol.
MEGA_PROTOCOL = "mega://"
MegaUriType = Literal["model", "dataset", "space", "kernel", "bucket"]
# Maps the plural MEGA URI prefix (e.g. ``datasets/``) to the canonical type.
MEGA_URI_TYPE_PREFIXES: dict[str, MegaUriType] = {
    "models": "model",
    "datasets": "dataset",
    "spaces": "space",
    "kernels": "kernel",
    "buckets": "bucket",
}


DiscussionTypeFilter = Literal["all", "discussion", "pull_request"]
DISCUSSION_TYPES: tuple[DiscussionTypeFilter, ...] = typing.get_args(DiscussionTypeFilter)
DiscussionStatusFilter = Literal["all", "open", "closed"]
DISCUSSION_STATUS: tuple[DiscussionTypeFilter, ...] = typing.get_args(DiscussionStatusFilter)

# Webhook subscription types
WEBHOOK_DOMAIN_T = Literal["repo", "discussions"]

# default cache
default_home = os.path.join(os.path.expanduser("~"), ".cache")
MEGA_HOME = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "MEGA_HOME",
            os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "mega"),
        )
    )
)

default_cache_path = os.path.join(MEGA_HOME, "hub")
default_assets_cache_path = os.path.join(MEGA_HOME, "assets")

MEGA_HUB_CACHE = os.path.expandvars(
    os.path.expanduser(
        os.getenv("MEGA_HUB_CACHE", default_cache_path)
    )
)
MEGA_ASSETS_CACHE = os.path.expandvars(
    os.path.expanduser(
        os.getenv("MEGA_ASSETS_CACHE", default_assets_cache_path)
    )
)

MEGA_HUB_OFFLINE = _is_true(os.environ.get("MEGA_HUB_OFFLINE"))


def is_offline_mode() -> bool:
    """Returns whether we are in offline mode for the Hub.

    When offline mode is enabled, all HTTP requests made with `get_session` will raise an `OfflineModeIsEnabled` exception.

    Example:
        ```py
        from megatensors._hub import is_offline_mode

        def list_files(repo_id: str):
            if is_offline_mode():
                ... # list files from local cache (degraded experience but still functional)
            else:
                ... # list files from Hub (complete experience)
        ```
    """
    return MEGA_HUB_OFFLINE


# File created to mark that the version check has been done.
# Check is performed once per 24 hours at most.
CHECK_FOR_UPDATE_DONE_PATH = os.path.join(MEGA_HOME, ".check_for_update_done")

# File caching the AI agent harnesses registry fetched from `{ENDPOINT}/api/agent-harnesses`.
# Refreshed once per 24 hours at most (see `utils/_detect_agent.py`).
AGENT_HARNESSES_PATH = os.path.join(MEGA_HOME, ".agent_harnesses.json")

# Set to skip the CLI update check (PyPI query + "new version available" warning at startup).
MEGA_HUB_DISABLE_UPDATE_CHECK = _is_true(os.environ.get("MEGA_HUB_DISABLE_UPDATE_CHECK"))

# If set, log level will be set to DEBUG and all requests made to the Hub will be logged
# as curl commands for reproducibility.
MEGA_DEBUG = _is_true(os.environ.get("MEGA_DEBUG"))

# Opt-out from telemetry requests
MEGA_HUB_DISABLE_TELEMETRY = (
    _is_true(os.environ.get("MEGA_HUB_DISABLE_TELEMETRY"))
    or _is_true(os.environ.get("DISABLE_TELEMETRY"))
    or _is_true(os.environ.get("DO_NOT_TRACK"))  # https://consoledonottrack.com/
)

MEGA_TOKEN_PATH = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "MEGA_TOKEN_PATH",
            os.path.join(MEGA_HOME, "token"),
        )
    )
)
MEGA_STORED_TOKENS_PATH = os.path.join(os.path.dirname(MEGA_TOKEN_PATH), "stored_tokens")

# Here, `True` will disable progress bars globally without possibility of enabling it
# programmatically. `False` will enable them without possibility of disabling them.
# If environment variable is not set (None), then the user is free to enable/disable
# them programmatically.
# TL;DR: env variable has priority over code
__MEGA_HUB_DISABLE_PROGRESS_BARS = os.environ.get("MEGA_HUB_DISABLE_PROGRESS_BARS")
MEGA_HUB_DISABLE_PROGRESS_BARS: bool | None = (
    _is_true(__MEGA_HUB_DISABLE_PROGRESS_BARS) if __MEGA_HUB_DISABLE_PROGRESS_BARS is not None else None
)

# Disable symlinks in the cache (files are copied instead of symlinked)
MEGA_HUB_DISABLE_SYMLINKS: bool = _is_true(os.environ.get("MEGA_HUB_DISABLE_SYMLINKS"))

# Disable warning on machines that do not support symlinks (e.g. Windows non-developer)
MEGA_HUB_DISABLE_SYMLINKS_WARNING: bool = _is_true(os.environ.get("MEGA_HUB_DISABLE_SYMLINKS_WARNING"))

# Disable warning when using experimental features
MEGA_HUB_DISABLE_EXPERIMENTAL_WARNING: bool = _is_true(os.environ.get("MEGA_HUB_DISABLE_EXPERIMENTAL_WARNING"))

# Disable sending the cached token by default is all HTTP requests to the Hub
MEGA_HUB_DISABLE_IMPLICIT_TOKEN: bool = _is_true(os.environ.get("MEGA_HUB_DISABLE_IMPLICIT_TOKEN"))

MEGA_XET_HIGH_PERFORMANCE: bool = _is_true(os.environ.get("MEGA_XET_HIGH_PERFORMANCE"))

# Bucket and mount path used when launching Jobs
MEGA_JOBS_ARTIFACTS_BUCKET_NAME: str = "jobs-artifacts"
MEGA_JOBS_ARTIFACTS_MOUNT_PATH: str = "/data"

# Used to override the etag timeout on a system level
MEGA_HUB_ETAG_TIMEOUT: int = _as_int(os.environ.get("MEGA_HUB_ETAG_TIMEOUT")) or DEFAULT_ETAG_TIMEOUT

# Used to override the get request timeout on a system level
# Also used as a default timeout for other requests if not specified (kept the naming for legacy reasons)
MEGA_HUB_DOWNLOAD_TIMEOUT: int = _as_int(os.environ.get("MEGA_HUB_DOWNLOAD_TIMEOUT")) or DEFAULT_DOWNLOAD_TIMEOUT

# Allows to add information about the requester in the user-agent (e.g. partner name)
MEGA_HUB_USER_AGENT_ORIGIN: str | None = os.environ.get("MEGA_HUB_USER_AGENT_ORIGIN")

# If OAuth didn't work after 2 redirects, there's likely a third-party cookie issue in the Space iframe view.
# In this case, we redirect the user to the non-iframe view.
OAUTH_MAX_REDIRECTS = 2

# OAuth-related environment variables injected by the Space
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
OAUTH_SCOPES = os.environ.get("OAUTH_SCOPES")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL")

# OAuth client ID of the Device Code login flow (RFC 8628) used by `mega auth login` / `login()`.
# Overridable for Hub deployments (staging, Enterprise) where the default client ID is not provisioned.
DEVICE_CODE_OAUTH_CLIENT_ID = os.environ.get("MEGA_DEVICE_CODE_OAUTH_CLIENT_ID", "mega-cli")

# Xet constants
MEGA_HEADER_X_XET_ENDPOINT = "X-Xet-Cas-Url"
MEGA_HEADER_X_XET_ACCESS_TOKEN = "X-Xet-Access-Token"
MEGA_HEADER_X_XET_EXPIRATION = "X-Xet-Token-Expiration"
MEGA_HEADER_X_XET_HASH = "X-Xet-Hash"
MEGA_HEADER_X_XET_REFRESH_ROUTE = "X-Xet-Refresh-Route"
MEGA_HEADER_LINK_XET_AUTH_KEY = "xet-auth"

default_xet_cache_path = os.path.join(MEGA_HOME, "xet")
MEGA_XET_CACHE = os.getenv("MEGA_XET_CACHE", default_xet_cache_path)
MEGA_HUB_DISABLE_XET: bool = _is_true(os.environ.get("MEGA_HUB_DISABLE_XET"))

# Bucket hosting the static sandbox server binary (see megatensors._hub.Sandbox)
SANDBOX_SERVER_BUCKET: str = "mega/sbx-server"
