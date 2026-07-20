from functools import lru_cache
from typing import Any, overload

from megatensors._hub import constants
from megatensors._hub.mega_api import InferenceProviderMapping
from megatensors._hub.inference._common import MimeBytes, RequestParameters, _as_dict
from megatensors._hub.inference._generated.types.chat_completion import ChatCompletionInputMessage
from megatensors._hub.utils import (
    build_mega_headers,
    get_session,
    get_token,
    logging,
    mega_raise_for_status,
)


logger = logging.get_logger(__name__)


# Dev purposes only.
# If you want to try to run inference for a new model locally before it's registered on huggingface.co
# for a given Inference Provider, you can add it to the following dictionary.
HARDCODED_MODEL_INFERENCE_MAPPING: dict[str, dict[str, InferenceProviderMapping]] = {
    # "MEGA model ID" => InferenceProviderMapping object initialized with "Model ID on Inference Provider's side"
    #
    # Example:
    # "Qwen/Qwen2.5-Coder-32B-Instruct": InferenceProviderMapping(model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    #                                    provider_id="Qwen2.5-Coder-32B-Instruct",
    #                                    task="conversational",
    #                                    status="live")
    "cerebras": {},
    "cohere": {},
    "deepinfra": {},
    "fal-ai": {},
    "fireworks-ai": {},
    "groq": {},
    "mega": {},
    "nscale": {},
    "ovhcloud": {},
    "replicate": {},
    "scaleway": {},
    "together": {},
    "wavespeed": {},
    "zai-org": {},
}


@overload
def filter_none(obj: dict[str, Any]) -> dict[str, Any]: ...
@overload
def filter_none(obj: list[Any]) -> list[Any]: ...


def filter_none(obj: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                v = filter_none(v)
            cleaned[k] = v
        return cleaned

    if isinstance(obj, list):
        return [filter_none(v) if isinstance(v, (dict, list)) else v for v in obj]

    raise ValueError(f"Expected dict or list, got {type(obj)}")


class TaskProviderHelper:
    """Base class for task-specific provider helpers."""

    def __init__(self, provider: str, base_url: str, task: str) -> None:
        self.provider = provider
        self.task = task
        self.base_url = base_url

    def prepare_request(
        self,
        *,
        inputs: Any,
        parameters: dict[str, Any],
        headers: dict,
        model: str | None,
        api_key: str | None,
        extra_payload: dict[str, Any] | None = None,
    ) -> RequestParameters:
        """
        Prepare the request to be sent to the provider.

        Each step (api_key, model, headers, url, payload) can be customized in subclasses.
        """
        # api_key from user, or local token, or raise error
        api_key = self._prepare_api_key(api_key)

        # Routed chat requests keep the canonical Hub model id. The MEGA Router
        # selects (or pins) the provider from the model suffix, so a MEGA token
        # must not require a separate provider-mapping lookup first. Direct
        # provider keys still use the provider's public model mapping below.
        if (
            api_key.startswith("mega_")
            and (
                self.task == "conversational"
                or self.task == "feature-extraction"
            )
            and model is not None
            and not model.startswith(("http://", "https://"))
        ):
            routed_model = _routed_model_id(model, self.provider)
            provider_mapping_info = InferenceProviderMapping(
                provider=self.provider,
                model_id=model,
                providerId=routed_model,
                status="live",
                task=self.task,
            )
        else:
            provider_mapping_info = self._prepare_mapping_info(model)

        # default HF headers + user headers (to customize in subclasses)
        headers = self._prepare_headers(headers, api_key)

        # routed URL if MEGA token, or direct URL (to customize in '_prepare_route' in subclasses)
        url = self._prepare_url(api_key, provider_mapping_info.provider_id)

        # prepare payload (to customize in subclasses)
        payload = self._prepare_payload_as_dict(inputs, parameters, provider_mapping_info=provider_mapping_info)
        if payload is not None:
            payload = recursive_merge(payload, filter_none(extra_payload or {}))

        # body data (to customize in subclasses)
        data = self._prepare_payload_as_bytes(inputs, parameters, provider_mapping_info, extra_payload)

        # check if both payload and data are set and return
        if payload is not None and data is not None:
            raise ValueError("Both payload and data cannot be set in the same request.")
        if payload is None and data is None:
            raise ValueError("Either payload or data must be set in the request.")

        # normalize headers to lowercase and add content-type if not present
        normalized_headers = self._normalize_headers(headers, payload, data)

        return RequestParameters(
            url=url,
            task=self.task,
            model=provider_mapping_info.provider_id,
            json=payload,
            data=data,
            headers=normalized_headers,
        )

    def get_response(
        self,
        response: bytes | dict,
        request_params: RequestParameters | None = None,
    ) -> Any:
        """
        Return the response in the expected format.

        Override this method in subclasses for customized response handling."""
        return response

    def _prepare_api_key(self, api_key: str | None) -> str:
        """Return the API key to use for the request.

        Usually not overwritten in subclasses."""
        if api_key is None:
            api_key = get_token()
        if api_key is None:
            raise ValueError(
                f"You must provide an api_key to work with {self.provider} API or log in with `mega auth login`."
            )
        return api_key

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        """Return the mapped model ID to use for the request.

        Usually not overwritten in subclasses."""
        if model is None:
            raise ValueError(f"Please provide an MEGA model ID supported by {self.provider}.")

        # hardcoded mapping for local testing
        if HARDCODED_MODEL_INFERENCE_MAPPING.get(self.provider, {}).get(model):
            return HARDCODED_MODEL_INFERENCE_MAPPING[self.provider][model]

        mappings = _fetch_inference_provider_mapping(model)
        provider_mapping = next(
            (
                mapping
                for mapping in mappings
                if mapping.provider == self.provider and mapping.task == self.task
            ),
            None,
        )

        if provider_mapping is None:
            supported_tasks = sorted(
                {mapping.task for mapping in mappings if mapping.provider == self.provider}
            )
            if supported_tasks:
                raise ValueError(
                    f"Model {model} is not supported for task {self.task} and provider {self.provider}. "
                    f"Supported tasks: {', '.join(supported_tasks)}."
                )
            raise ValueError(f"Model {model} is not supported by provider {self.provider}.")

        if provider_mapping.status == "staging":
            logger.warning(
                f"Model {model} is in staging mode for provider {self.provider}. Meant for test purposes only."
            )
        if provider_mapping.status == "error":
            logger.warning(
                f"Our latest automated health check on model '{model}' for provider '{self.provider}' did not complete successfully.  "
                "Inference call might fail."
            )
        return provider_mapping

    def _normalize_headers(
        self, headers: dict[str, Any], payload: dict[str, Any] | None, data: MimeBytes | None
    ) -> dict[str, Any]:
        """Normalize the headers to use for the request.

        Override this method in subclasses for customized headers.
        """
        normalized_headers = {key.lower(): value for key, value in headers.items() if value is not None}
        if normalized_headers.get("content-type") is None:
            if data is not None and data.mime_type is not None:
                normalized_headers["content-type"] = data.mime_type
            elif payload is not None:
                normalized_headers["content-type"] = "application/json"
        return normalized_headers

    def _prepare_headers(self, headers: dict, api_key: str) -> dict[str, Any]:
        """Return the headers to use for the request.

        Override this method in subclasses for customized headers.
        """
        return {**build_mega_headers(token=api_key), **headers}

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        """Return the URL to use for the request.

        Usually not overwritten in subclasses."""
        if api_key.startswith("mega_"):
            routed_path = {
                "conversational": "/v1/chat/completions",
                "feature-extraction": "/v1/embeddings",
            }.get(self.task)
            if routed_path is not None:
                return f"{constants.INFERENCE_ROUTER_ENDPOINT}{routed_path}"
        base_url = self._prepare_base_url(api_key)
        route = self._prepare_route(mapped_model, api_key)
        return f"{base_url.rstrip('/')}/{route.lstrip('/')}"

    def _prepare_base_url(self, api_key: str) -> str:
        """Return the base URL to use for the request.

        Usually not overwritten in subclasses."""
        # Route MEGA tokens through the public data plane. Provider API keys are
        # sent directly to that provider's published endpoint.
        if api_key.startswith("mega_"):
            logger.info(f"Calling '{self.provider}' provider through MEGA router.")
            return constants.INFERENCE_ROUTER_ENDPOINT
        else:
            logger.info(f"Calling '{self.provider}' provider directly.")
            return self.base_url

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        """Return the route to use for the request.

        Override this method in subclasses for customized routes.
        """
        return ""

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        """Return the payload to use for the request, as a dict.

        Override this method in subclasses for customized payloads.
        Only one of `_prepare_payload_as_dict` and `_prepare_payload_as_bytes` should return a value.
        """
        return None

    def _prepare_payload_as_bytes(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
        extra_payload: dict | None,
    ) -> MimeBytes | None:
        """Return the body to use for the request, as bytes.

        Override this method in subclasses for customized body data.
        Only one of `_prepare_payload_as_dict` and `_prepare_payload_as_bytes` should return a value.
        """
        return None


class BaseConversationalTask(TaskProviderHelper):
    """
    Base class for conversational (chat completion) tasks.
    The schema follows the OpenAI API format defined here: https://platform.openai.com/docs/api-reference/chat
    """

    def __init__(self, provider: str, base_url: str):
        super().__init__(provider=provider, base_url=base_url, task="conversational")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/chat/completions"

    def _prepare_payload_as_dict(
        self,
        inputs: list[dict | ChatCompletionInputMessage],
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
    ) -> dict | None:
        return filter_none({"messages": inputs, **parameters, "model": provider_mapping_info.provider_id})


class AutoRouterConversationalTask(BaseConversationalTask):
    """
    Auto-router for conversational tasks.

    We let the MEGA router select the best provider for the model, based on availability and user preferences.
    This is a special case since the selection is done server-side (avoid 1 API call to fetch provider mapping).
    """

    def __init__(self, policy: str = "auto"):
        if policy not in ("auto", "fastest", "cheapest", "preferred"):
            raise ValueError(f"Unsupported MEGA routing policy: {policy}")
        self.policy = policy
        super().__init__(provider=policy, base_url=constants.INFERENCE_ROUTER_ENDPOINT)

    def _prepare_base_url(self, api_key: str) -> str:
        """Return the base URL to use for the request.

        Usually not overwritten in subclasses."""
        # Automatic routing requires a MEGA token because the Router performs
        # Hub token introspection, preauthorization, and provider selection.
        if not api_key.startswith("mega_"):
            raise ValueError("Cannot select auto-router when using non-MEGA API key.")
        else:
            return self.base_url  # No `/auto` suffix in the URL

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        """
        In auto-router, we don't need to fetch provider mapping info.
        We just return a dummy mapping info with provider_id set to the MEGA model ID.
        """
        if model is None:
            raise ValueError("Please provide an MEGA model ID.")

        return InferenceProviderMapping(
            provider=self.policy,
            model_id=model,
            providerId=_routed_model_id(model, self.policy),
            status="live",
            task="conversational",
        )


def _routed_model_id(model: str, provider_or_policy: str) -> str:
    """Build the Router's ``owner/model[:selection]`` identifier."""
    slash = model.find("/")
    colon = model.rfind(":")
    if slash <= 0:
        raise ValueError("Inference model must use owner/model format.")
    if colon > slash:
        existing = model[colon + 1 :].lower()
        requested = provider_or_policy.lower()
        if requested in ("auto", "fastest") or existing == requested:
            return model
        raise ValueError(
            f"Model '{model}' already selects '{existing}', which conflicts with provider '{provider_or_policy}'."
        )
    if provider_or_policy in ("auto", "fastest"):
        return model
    return f"{model}:{provider_or_policy}"


class AutoRouterFeatureExtractionTask(TaskProviderHelper):
    """OpenAI embeddings adapter backed by MEGA server-side routing."""

    def __init__(self, policy: str = "auto"):
        if policy not in ("auto", "fastest", "cheapest", "preferred"):
            raise ValueError(f"Unsupported MEGA routing policy: {policy}")
        self.policy = policy
        super().__init__(
            provider=policy,
            base_url=constants.INFERENCE_ROUTER_ENDPOINT,
            task="feature-extraction",
        )

    def _prepare_base_url(self, api_key: str) -> str:
        if not api_key.startswith("mega_"):
            raise ValueError("Cannot select auto-router when using non-MEGA API key.")
        return self.base_url

    def _prepare_mapping_info(self, model: str | None) -> InferenceProviderMapping:
        if model is None:
            raise ValueError("Please provide an MEGA model ID.")
        return InferenceProviderMapping(
            provider=self.policy,
            model_id=model,
            providerId=_routed_model_id(model, self.policy),
            status="live",
            task="feature-extraction",
        )

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/embeddings"

    def _prepare_payload_as_dict(
        self,
        inputs: Any,
        parameters: dict,
        provider_mapping_info: InferenceProviderMapping,
    ) -> dict | None:
        return filter_none(
            {
                "input": inputs,
                "model": provider_mapping_info.provider_id,
                "dimensions": parameters.get("dimensions"),
                "encoding_format": parameters.get("encoding_format") or "float",
            }
        )

    def get_response(self, response: bytes | dict, request_params: RequestParameters | None = None) -> Any:
        records = _as_dict(response).get("data")
        if not isinstance(records, list):
            raise ValueError("MEGA embeddings response is missing data.")
        ordered = sorted(
            (item for item in records if isinstance(item, dict)),
            key=lambda item: int(item.get("index", 0)),
        )
        return [item.get("embedding") for item in ordered]


class BaseTextGenerationTask(TaskProviderHelper):
    """
    Base class for text-generation (completion) tasks.
    The schema follows the OpenAI API format defined here: https://platform.openai.com/docs/api-reference/completions
    """

    def __init__(self, provider: str, base_url: str):
        super().__init__(provider=provider, base_url=base_url, task="text-generation")

    def _prepare_route(self, mapped_model: str, api_key: str) -> str:
        return "/v1/completions"

    def _prepare_payload_as_dict(
        self, inputs: Any, parameters: dict, provider_mapping_info: InferenceProviderMapping
    ) -> dict | None:
        return filter_none({"prompt": inputs, **parameters, "model": provider_mapping_info.provider_id})


@lru_cache(maxsize=None)
def _fetch_inference_provider_mapping(model: str) -> list["InferenceProviderMapping"]:
    """
    Fetch live provider mappings for a model from the public Hub catalog.
    """
    response = get_session().get(
        constants.INFERENCE_MODELS_ENDPOINT,
        headers=build_mega_headers(token=False),
    )
    mega_raise_for_status(response, endpoint_name="inference model catalog")
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("The inference model catalog returned an invalid response.")

    model_entry = next(
        (entry for entry in models if isinstance(entry, dict) and entry.get("id") == model),
        None,
    )
    providers = model_entry.get("providers") if model_entry is not None else None
    if not isinstance(providers, list):
        raise ValueError(f"No provider mapping found for model {model}")

    task_names = {
        "chat-completions": "conversational",
        "embeddings": "feature-extraction",
    }
    mappings: list[InferenceProviderMapping] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        provider_name = provider.get("provider")
        provider_model = provider.get("provider_model")
        task = provider.get("task")
        status = provider.get("status")
        if not all(isinstance(value, str) and value for value in (provider_name, provider_model, task)):
            continue
        mappings.append(
            InferenceProviderMapping(
                provider=provider_name,
                model_id=model,
                providerId=provider_model,
                status=status if status in ("error", "live", "staging") else "live",
                task=task_names.get(task, task),
            )
        )
    if not mappings:
        raise ValueError(f"No provider mapping found for model {model}")
    return mappings


def recursive_merge(dict1: dict, dict2: dict) -> dict:
    return {
        **dict1,
        **{
            key: recursive_merge(dict1[key], value)
            if (key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict))
            else value
            for key, value in dict2.items()
        },
    }
