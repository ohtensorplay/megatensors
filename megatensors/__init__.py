# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

__version__ = version(__name__)

from .api import (
    append_footer_overlay,
    iter_tensors,
    load_kv_tensor,
    load_model,
    load_state_dict,
    load_tensor,
    load_tokenizer,
    open_kv_cache,
    write_kv_cache,
)
from .common import (
    MegaKvMetadata,
    MegaModelInfo,
    MegaTensorsMetadata,
    MegaTokenizerInfo,
    TensorFrame,
)
from .convert import ConvertResult, convert_model, resolve_artifacts
from .hub import (
    AccountKeyInfo,
    CommunityAuthor,
    CreatedWebhook,
    DiscussionInfo,
    DiscussionMessageInfo,
    DiscussionPage,
    DiscussionPermissions,
    DiscussionThread,
    MegaHubClient,
    MegaHubError,
    PullRequestInfo,
    WebhookDeliveryInfo,
    WebhookInfo,
    WebhookLastDeliveryInfo,
)
from .mega_hub import (
    HfApi,
    HfFileSystem,
    MegaApi,
    MegaFileSystem,
    hf_hub_download,
    hf_hub_url,
    mega_hub_download,
    mega_hub_url,
    snapshot_download,
)
from ._hub import AsyncInferenceClient, InferenceClient
from .loader import mega_open
from .signing import SigningConfig, sign_artifact

__all__ = [
    "__version__",
    "mega_open",
    "iter_tensors",
    "load_tensor",
    "load_state_dict",
    "load_model",
    "load_tokenizer",
    "open_kv_cache",
    "load_kv_tensor",
    "write_kv_cache",
    "append_footer_overlay",
    "convert_model",
    "resolve_artifacts",
    "ConvertResult",
    "MegaHubClient",
    "MegaHubError",
    "AccountKeyInfo",
    "CommunityAuthor",
    "CreatedWebhook",
    "DiscussionInfo",
    "DiscussionMessageInfo",
    "DiscussionPage",
    "DiscussionPermissions",
    "DiscussionThread",
    "PullRequestInfo",
    "WebhookDeliveryInfo",
    "WebhookInfo",
    "WebhookLastDeliveryInfo",
    "MegaApi",
    "HfApi",
    "InferenceClient",
    "AsyncInferenceClient",
    "MegaFileSystem",
    "HfFileSystem",
    "mega_hub_download",
    "mega_hub_url",
    "hf_hub_download",
    "hf_hub_url",
    "snapshot_download",
    "SigningConfig",
    "sign_artifact",
    "MegaKvMetadata",
    "MegaModelInfo",
    "MegaTensorsMetadata",
    "MegaTokenizerInfo",
    "TensorFrame",
]
