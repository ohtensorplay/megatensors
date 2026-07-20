from typing import Literal, Union

from megatensors._hub.inference._providers.featherless_ai import (
    FeatherlessConversationalTask,
    FeatherlessTextGenerationTask,
)
from megatensors._hub.utils import logging

from ._common import (
    AutoRouterConversationalTask,
    AutoRouterFeatureExtractionTask,
    TaskProviderHelper,
    _fetch_inference_provider_mapping,
)
from .cerebras import CerebrasConversationalTask
from .cohere import CohereConversationalTask
from .deepinfra import (
    DeepInfraAutomaticSpeechRecognitionTask,
    DeepInfraConversationalTask,
    DeepInfraTextGenerationTask,
)
from .fal_ai import (
    FalAIAutomaticSpeechRecognitionTask,
    FalAIImageSegmentationTask,
    FalAIImageToImageTask,
    FalAIImageToVideoTask,
    FalAITextToImageTask,
    FalAITextToSpeechTask,
    FalAITextToVideoTask,
)
from .fireworks_ai import FireworksAIConversationalTask
from .groq import GroqConversationalTask
from .mega_inference import (
    MegaInferenceBinaryInputTask,
    MegaInferenceConversational,
    MegaInferenceFeatureExtractionTask,
    MegaInferenceTask,
)
from .novita import NovitaConversationalTask, NovitaTextGenerationTask, NovitaTextToVideoTask
from .nscale import NscaleConversationalTask, NscaleTextToImageTask
from .openai import OpenAIConversationalTask
from .ovhcloud import OVHcloudConversationalTask
from .publicai import PublicAIConversationalTask
from .replicate import (
    ReplicateAutomaticSpeechRecognitionTask,
    ReplicateImageToImageTask,
    ReplicateTask,
    ReplicateTextToImageTask,
    ReplicateTextToSpeechTask,
)
from .scaleway import ScalewayConversationalTask, ScalewayFeatureExtractionTask
from .together import (
    TogetherConversationalTask,
    TogetherFeatureExtractionTask,
    TogetherImageToImageTask,
    TogetherImageToVideoTask,
    TogetherTextGenerationTask,
    TogetherTextToImageTask,
    TogetherTextToSpeechTask,
    TogetherTextToVideoTask,
)
from .wavespeed import (
    WavespeedAIImageToImageTask,
    WavespeedAIImageToVideoTask,
    WavespeedAITextToImageTask,
    WavespeedAITextToVideoTask,
)
from .zai_org import ZaiConversationalTask, ZaiTextToImageTask


logger = logging.get_logger(__name__)


PROVIDER_T = Literal[
    "cerebras",
    "cohere",
    "deepinfra",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "mega",
    "novita",
    "nscale",
    "openai",
    "ovhcloud",
    "publicai",
    "replicate",
    "scaleway",
    "together",
    "wavespeed",
    "zai-org",
]

PROVIDER_OR_POLICY_T = Union[
    PROVIDER_T,
    Literal["auto", "fastest", "cheapest", "preferred"],
]

CONVERSATIONAL_ROUTERS = {
    policy: AutoRouterConversationalTask(policy)
    for policy in ("auto", "fastest", "cheapest", "preferred")
}
EMBEDDING_ROUTERS = {
    policy: AutoRouterFeatureExtractionTask(policy)
    for policy in ("auto", "fastest", "cheapest", "preferred")
}

PROVIDERS: dict[PROVIDER_T, dict[str, TaskProviderHelper]] = {
    "cerebras": {
        "conversational": CerebrasConversationalTask(),
    },
    "cohere": {
        "conversational": CohereConversationalTask(),
    },
    "deepinfra": {
        "automatic-speech-recognition": DeepInfraAutomaticSpeechRecognitionTask(),
        "conversational": DeepInfraConversationalTask(),
        "text-generation": DeepInfraTextGenerationTask(),
    },
    "fal-ai": {
        "automatic-speech-recognition": FalAIAutomaticSpeechRecognitionTask(),
        "text-to-image": FalAITextToImageTask(),
        "text-to-speech": FalAITextToSpeechTask(),
        "text-to-video": FalAITextToVideoTask(),
        "image-to-video": FalAIImageToVideoTask(),
        "image-to-image": FalAIImageToImageTask(),
        "image-segmentation": FalAIImageSegmentationTask(),
    },
    "featherless-ai": {
        "conversational": FeatherlessConversationalTask(),
        "text-generation": FeatherlessTextGenerationTask(),
    },
    "fireworks-ai": {
        "conversational": FireworksAIConversationalTask(),
    },
    "groq": {
        "conversational": GroqConversationalTask(),
    },
    "mega": {
        "text-to-image": MegaInferenceTask("text-to-image"),
        "conversational": MegaInferenceConversational(),
        "text-generation": MegaInferenceTask("text-generation"),
        "text-classification": MegaInferenceTask("text-classification"),
        "question-answering": MegaInferenceTask("question-answering"),
        "audio-classification": MegaInferenceBinaryInputTask("audio-classification"),
        "automatic-speech-recognition": MegaInferenceBinaryInputTask("automatic-speech-recognition"),
        "fill-mask": MegaInferenceTask("fill-mask"),
        "feature-extraction": MegaInferenceFeatureExtractionTask(),
        "image-classification": MegaInferenceBinaryInputTask("image-classification"),
        "image-segmentation": MegaInferenceBinaryInputTask("image-segmentation"),
        "document-question-answering": MegaInferenceTask("document-question-answering"),
        "image-to-text": MegaInferenceBinaryInputTask("image-to-text"),
        "object-detection": MegaInferenceBinaryInputTask("object-detection"),
        "audio-to-audio": MegaInferenceBinaryInputTask("audio-to-audio"),
        "zero-shot-image-classification": MegaInferenceBinaryInputTask("zero-shot-image-classification"),
        "zero-shot-classification": MegaInferenceTask("zero-shot-classification"),
        "image-to-image": MegaInferenceBinaryInputTask("image-to-image"),
        "sentence-similarity": MegaInferenceTask("sentence-similarity"),
        "table-question-answering": MegaInferenceTask("table-question-answering"),
        "tabular-classification": MegaInferenceTask("tabular-classification"),
        "text-to-speech": MegaInferenceTask("text-to-speech"),
        "token-classification": MegaInferenceTask("token-classification"),
        "translation": MegaInferenceTask("translation"),
        "summarization": MegaInferenceTask("summarization"),
        "visual-question-answering": MegaInferenceBinaryInputTask("visual-question-answering"),
    },
    "novita": {
        "text-generation": NovitaTextGenerationTask(),
        "conversational": NovitaConversationalTask(),
        "text-to-video": NovitaTextToVideoTask(),
    },
    "nscale": {
        "conversational": NscaleConversationalTask(),
        "text-to-image": NscaleTextToImageTask(),
    },
    "openai": {
        "conversational": OpenAIConversationalTask(),
    },
    "ovhcloud": {
        "conversational": OVHcloudConversationalTask(),
    },
    "publicai": {
        "conversational": PublicAIConversationalTask(),
    },
    "replicate": {
        "automatic-speech-recognition": ReplicateAutomaticSpeechRecognitionTask(),
        "image-to-image": ReplicateImageToImageTask(),
        "text-to-image": ReplicateTextToImageTask(),
        "text-to-speech": ReplicateTextToSpeechTask(),
        "text-to-video": ReplicateTask("text-to-video"),
    },
    "scaleway": {
        "conversational": ScalewayConversationalTask(),
        "feature-extraction": ScalewayFeatureExtractionTask(),
    },
    "together": {
        "conversational": TogetherConversationalTask(),
        "feature-extraction": TogetherFeatureExtractionTask(),
        "image-to-image": TogetherImageToImageTask(),
        "image-to-video": TogetherImageToVideoTask(),
        "text-generation": TogetherTextGenerationTask(),
        "text-to-image": TogetherTextToImageTask(),
        "text-to-speech": TogetherTextToSpeechTask(),
        "text-to-video": TogetherTextToVideoTask(),
    },
    "wavespeed": {
        "text-to-image": WavespeedAITextToImageTask(),
        "text-to-video": WavespeedAITextToVideoTask(),
        "image-to-image": WavespeedAIImageToImageTask(),
        "image-to-video": WavespeedAIImageToVideoTask(),
    },
    "zai-org": {
        "conversational": ZaiConversationalTask(),
        "text-to-image": ZaiTextToImageTask(),
    },
}


def get_provider_helper(provider: PROVIDER_OR_POLICY_T | None, task: str, model: str | None) -> TaskProviderHelper:
    """Get provider helper instance by name and task.

    Args:
        provider (`str`, *optional*): name of the provider, or "auto" to automatically select the provider for the model.
        task (`str`): Name of the task
        model (`str`, *optional*): Name of the model
    Returns:
        TaskProviderHelper: Helper instance for the specified provider and task

    Raises:
        ValueError: If provider or task is not supported
    """

    if (model is None and provider in (None, "auto", "fastest", "cheapest", "preferred")) or (
        model is not None and model.startswith(("http://", "https://"))
    ):
        provider = "mega"

    if provider is None:
        logger.info(
            "No provider specified for task `conversational`. Defaulting to server-side auto routing."
            if task == "conversational"
            else "Defaulting to 'auto' which will select the first provider available for the model, sorted by the user's order in https://mega.tensorplay.cn/settings/inference-providers."
        )
        provider = "auto"

    if provider in ("auto", "fastest", "cheapest", "preferred"):
        if model is None:
            raise ValueError(f"Specifying a model is required when provider is '{provider}'")
        if task == "conversational":
            # Special case: we have a dedicated auto-router for conversational models. No need to fetch provider mapping.
            return CONVERSATIONAL_ROUTERS[provider]

        if task == "feature-extraction":
            return EMBEDDING_ROUTERS[provider]

        if provider != "auto":
            raise ValueError(f"Routing policy '{provider}' is currently supported for conversational models only.")

        provider_mapping = _fetch_inference_provider_mapping(model)
        provider = next(iter(provider_mapping)).provider

    provider_tasks = PROVIDERS.get(provider)  # type: ignore
    if provider_tasks is None:
        raise ValueError(
            f"Provider '{provider}' not supported. Available values: 'auto' or any provider from {list(PROVIDERS.keys())}."
            "Passing 'auto' (default value) will automatically select the first provider available for the model, sorted "
            "by the user's order in https://mega.tensorplay.cn/settings/inference-providers."
        )

    if task not in provider_tasks:
        raise ValueError(
            f"Task '{task}' not supported for provider '{provider}'. Available tasks: {list(provider_tasks.keys())}"
        )
    return provider_tasks[task]
