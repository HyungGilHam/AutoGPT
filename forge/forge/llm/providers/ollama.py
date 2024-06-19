from __future__ import annotations

import enum
import logging
from typing import Any, Optional

import tiktoken
from pydantic import SecretStr

from forge.models.config import UserConfigurable

from ._openai_base import BaseOpenAIChatProvider
from .schema import (
    ChatModelInfo,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)

from .schema import (
    ChatModelInfo,
    EmbeddingModelInfo,
    ModelProviderService,
    _ModelName,
)

from typing import (
    Any,
    Optional,
    Sequence,
)

class OllamaModelName(str, enum.Enum):
    LLAMA3_8B = "llama3-8b-ollama"
    LLAMA3_70B = "llama3-70b-ollama"

OLLAMA_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=OllamaModelName.LLAMA3_8B,
            provider_name=ModelProviderName.OLLAMA,
            prompt_token_cost=0.05 / 1e6,
            completion_token_cost=0.10 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OllamaModelName.LLAMA3_70B,
            provider_name=ModelProviderName.OLLAMA,
            prompt_token_cost=0.59 / 1e6,
            completion_token_cost=0.79 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
    ]
}

class OllamaCredentials(ModelProviderCredentials):
    """Credentials for Ollama."""
    api_key: SecretStr = UserConfigurable(from_env="OLLAMA_API_KEY")  # type: ignore
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OLLAMA_API_BASE_URL"
    )

    def get_api_access_kwargs(self) -> dict[str, str]:
        return {
            k: v.get_secret_value()
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }.items()
            if v is not None
        }


class OllamaSettings(ModelProviderSettings):
    credentials: Optional[OllamaCredentials]  # type: ignore
    budget: ModelProviderBudget  # type: ignore

class OllamaProvider(BaseOpenAIChatProvider[OllamaModelName, OllamaSettings]):
    CHAT_MODELS = OLLAMA_CHAT_MODELS
    MODELS = CHAT_MODELS

    default_settings = OllamaSettings(
        name="ollama_provider",
        description="Provides access to ollama API.",
        configuration=ModelProviderConfiguration(),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: OllamaSettings
    _configuration: ModelProviderConfiguration
    _credentials: OllamaCredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[OllamaSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(OllamaProvider, self).__init__(settings=settings, logger=logger)

    async def get_available_models(
        self,
    ) -> Sequence[ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]]:
        # 직접 입력하는 방식으로 수정된 메서드
        _models = [
             ChatModelInfo(
            name=OllamaModelName.LLAMA3_8B,
            provider_name=ModelProviderName.OLLAMA,
            prompt_token_cost=0.05 / 1e6,
            completion_token_cost=0.10 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OllamaModelName.LLAMA3_70B,
            provider_name=ModelProviderName.OLLAMA,
            prompt_token_cost=0.59 / 1e6,
            completion_token_cost=0.79 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        )
        ]
        return _models
    
    def get_tokenizer(self, model_name: OllamaModelName) -> ModelTokenizer[Any]:
        # HACK: No official tokenizer is available for Groq
        return tiktoken.encoding_for_model("gpt-3.5-turbo")