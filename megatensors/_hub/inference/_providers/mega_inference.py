import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from megatensors._hub import constants
from megatensors._hub.mega_api import InferenceProviderMapping
from megatensors._hub.inference._common import (
    MimeBytes,
    RequestParameters,
    _b64_encode,
    _bytes_to_dict,
    _open_as_mime_bytes,
)
from megatensors._hub.inference._providers._common import TaskProviderHelper, filter_none
from megatensors._hub.utils import build_mega_headers, get_session, get_token, mega_raise_for_status


class MegaInferenceTask(TaskProviderHelper):
    """Base class for MEGA Inference API tasks."""

    def __init__(self, task: str):
        super().__init__(
            provider="mega",
            base_url=constants.INFERENCE_PROXY_TEMPLATE.format(provider="mega"),
            task=task,
        )

    def _prepare_api_key(self, api_key: str | None) -> str:
        # special case: for MEGA Inference we allow not providing an API key
        return api_key or get_token()  # type: ignore

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        if model is not None and model.startswith(("http://", "https://")):
            return InferenceProviderMapping(
                provider="mega", providerId=model, model_id=model, task=self.task, status="live"
            )
        model_id = model if model is not None else _fetch_recommended_models().get(self.task)
        if model_id is None:
            raise ValueError(
                f"Task {self.task} has no recommended model for MEGA Inference. Please specify a model"
                " explicitly. Visit https://mega.tensorplay.cn/tasks for more info."
            )
        _check_supported_task(model_id, self.task)
        return InferenceProviderMapping(
            provider="mega", providerId=model_id, model_id=model_id, task=self.task, status="live"
        )

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        # mega provider can handle URLs (e.g. Inference Endpoints or TGI deployment)
        if mapped_model.startswith(("http://", "https://")):
            return mapped_model
        return (
            # Feature-extraction and sentence-similarity are the only cases where we handle models with several tasks.
            f"{self.base_url}/models/{mapped_model}/pipeline/{self.task}"
            if self.task in ("feature-extraction", "sentence-similarity")
            # Otherwise, we use the default endpoint
            else f"{self.base_url}/models/{mapped_model}"
        )

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        if isinstance(inputs, bytes):
            raise ValueError(f"Unexpected binary input for task {self.task}.")
        if isinstance(inputs, Path):
            raise ValueError(f"Unexpected path input for task {self.task} (got {inputs})")
        return filter_none({"inputs": inputs, "parameters": parameters})


class MegaInferenceBinaryInputTask(MegaInferenceTask):
    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return None

    def _prepare_payload_as_bytes(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
        extra_payload: dict | None,
    ) -> MimeBytes | None:
        parameters = filter_none(parameters)
        extra_payload = extra_payload or {}
        has_parameters = len(parameters) > 0 or len(extra_payload) > 0

        # Raise if not a binary object or a local path or a URL.
        if not isinstance(inputs, (bytes, Path)) and not isinstance(inputs, str):
            raise ValueError(f"Expected binary inputs or a local path or a URL. Got {inputs}")

        # Send inputs as raw content when no parameters are provided
        if not has_parameters:
            return _open_as_mime_bytes(inputs)

        # Otherwise encode as b64
        return MimeBytes(
            json.dumps({"inputs": _b64_encode(inputs), "parameters": parameters, **extra_payload}).encode("utf-8"),
            mime_type="application/json",
        )


class MegaInferenceConversational(MegaInferenceTask):
    def __init__(self):
        super().__init__("conversational")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        payload = filter_none(parameters)
        mapped_model = provider_mapping_info.provider_id
        payload_model = parameters.get("model") or mapped_model

        if payload_model is None or payload_model.startswith(("http://", "https://")):
            payload_model = "dummy"

        response_format = parameters.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            payload["response_format"] = {
                "type": "json_object",
                "value": response_format["json_schema"]["schema"],
            }
        return {**payload, "model": payload_model, "messages": inputs}

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        base_url = (
            mapped_model
            if mapped_model.startswith(("http://", "https://"))
            else constants.INFERENCE_ROUTER_ENDPOINT
        )
        return _build_chat_completion_url(base_url)


def _build_chat_completion_url(model_url: str) -> str:
    parsed = urlparse(model_url)
    path = parsed.path.rstrip("/")

    # If the path already ends with /chat/completions, we're done!
    if path.endswith("/chat/completions"):
        return model_url

    # Append /chat/completions if not already present
    if path.endswith("/v1"):
        new_path = path + "/chat/completions"
    # If path was empty or just "/", set the full path
    elif not path:
        new_path = "/v1/chat/completions"
    # Append /v1/chat/completions if not already present
    else:
        new_path = path + "/v1/chat/completions"

    # Reconstruct the URL with the new path and original query parameters.
    new_parsed = parsed._replace(path=new_path)
    return str(urlunparse(new_parsed))


@lru_cache(maxsize=1)
def _fetch_recommended_models() -> dict[str, str | None]:
    response = get_session().get(f"{constants.ENDPOINT}/api/tasks", headers=build_mega_headers())
    mega_raise_for_status(response)
    return {task: next(iter(details["widgetModels"]), None) for task, details in response.json().items()}


@lru_cache(maxsize=None)
def _check_supported_task(model: str, task: str) -> None:
    from megatensors._hub.mega_api import MegaApi

    model_info = MegaApi().model_info(model)
    pipeline_tag = model_info.pipeline_tag
    tags = model_info.tags or []
    is_conversational = "conversational" in tags
    if task in ("text-generation", "conversational"):
        if pipeline_tag == "text-generation":
            # text-generation + conversational tag -> both tasks allowed
            if is_conversational:
                return
            # text-generation without conversational tag -> only text-generation allowed
            if task == "text-generation":
                return
            raise ValueError(f"Model '{model}' doesn't support task '{task}'.")

    if pipeline_tag == "text2text-generation":
        if task == "text-generation":
            return
        raise ValueError(f"Model '{model}' doesn't support task '{task}'.")

    if pipeline_tag == "image-text-to-text":
        if is_conversational and task == "conversational":
            return  # Only conversational allowed if tagged as conversational
        raise ValueError("Non-conversational image-text-to-text task is not supported.")

    if (
        task in ("feature-extraction", "sentence-similarity")
        and pipeline_tag in ("feature-extraction", "sentence-similarity")
        and task in tags
    ):
        # feature-extraction and sentence-similarity are interchangeable for MEGA Inference
        return

    # For all other tasks, just check pipeline tag
    if pipeline_tag != task:
        raise ValueError(
            f"Model '{model}' doesn't support task '{task}'. Supported tasks: '{pipeline_tag}', got: '{task}'"
        )
    return


class MegaInferenceFeatureExtractionTask(MegaInferenceTask):
    def __init__(self):
        super().__init__("feature-extraction")

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        if isinstance(inputs, bytes):
            raise ValueError(f"Unexpected binary input for task {self.task}.")
        if isinstance(inputs, Path):
            raise ValueError(f"Unexpected path input for task {self.task} (got {inputs})")

        # The public data plane intentionally exposes the OpenAI embeddings
        # contract rather than the legacy /models/{id} feature-extraction body.
        return filter_none(
            {
                "input": inputs,
                "model": provider_mapping_info.provider_id,
                "dimensions": parameters.get("dimensions"),
                "encoding_format": parameters.get("encoding_format") or "float",
            }
        )

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        if mapped_model.startswith(("http://", "https://")):
            return mapped_model
        return f"{constants.INFERENCE_ROUTER_ENDPOINT}/v1/embeddings"

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        if isinstance(response, bytes):
            response = _bytes_to_dict(response)
        if isinstance(response, dict) and isinstance(response.get("data"), list):
            records = sorted(
                (item for item in response["data"] if isinstance(item, dict)),
                key=lambda item: int(item.get("index", 0)),
            )
            return [item.get("embedding") for item in records]
        return response
