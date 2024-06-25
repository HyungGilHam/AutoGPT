from __future__ import annotations

import json
import enum
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
)
from forge.json.parsing import extract_dict_from_json
from forge.utils.exceptions import InvalidAgentResponseError
from forge.models.utils import ModelWithSummary  # 추가된 부분

import tiktoken
from pydantic import SecretStr, BaseModel
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

from forge.models.config import UserConfigurable

from ._openai_base import BaseOpenAIChatProvider
from .schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    BaseChatModelProvider,
    BaseEmbeddingModelProvider,
    BaseModelProvider,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    ModelProviderService,
    _ModelName,
    _ModelProviderSettings,
)

from .schema import (
    ChatModelInfo,
    EmbeddingModelInfo,
    ModelProviderService,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderName,
    ModelProviderCredentials,
    ModelProviderSettings,
    ModelTokenizer,
    _ModelName,
)

from typing import (
    Any,
    Optional,
    Sequence,
)


_T = TypeVar("_T")

class OllamaModelName(str, enum.Enum):
    LLAMA3_8B = "llama3"
    LLAMA3_70B = "llama3"

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

class AsyncOllamaChat:
    def __init__(self, client: httpx.AsyncClient, base_url: str):
        self.client = client
        self.base_url = base_url

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def create(self, model: str, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {"model": model, "prompt": prompt}
        response_text = ""
        async with self.client.stream("POST", url, headers=headers, json=data) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    response_json = httpx.Response(200, content=line).json()
                    response_text += response_json["response"]
                    if response_json.get("done", False):
                        break
        return response_text

class AsyncOllama:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.chat = AsyncOllamaChat(self.client, base_url)

    async def close(self):
        await self.client.aclose()

from pydantic import Field
class AssistantThoughts(ModelWithSummary):
    observations: str = Field(
        ..., description="Relevant observations from your last action (if any)"
    )
    text: str = Field(..., description="Thoughts")
    reasoning: str = Field(..., description="Reasoning behind the thoughts")
    self_criticism: str = Field(..., description="Constructive self-criticism")
    plan: list[str] = Field(
        ..., description="Short list that conveys the long-term plan"
    )
    speak: str = Field(..., description="Summary of thoughts, to say to user")

    def summary(self) -> str:
        return self.text

class UseTool(BaseModel):
    name: str
    arguments: dict

class ActionProposal(BaseModel):
    thoughts: str | ModelWithSummary
    use_tool: AssistantFunctionCall

class OneShotAgentActionProposal(ActionProposal):
    thoughts: AssistantThoughts
    
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

        self._client = AsyncOllama(
            api_key=self._credentials.api_key.get_secret_value(),
            base_url=self._credentials.api_base.get_secret_value()
        )

    async def get_available_models(
        self,
    ) -> Sequence[ChatModelInfo[OllamaModelName]]:
        return list(self.MODELS.values())

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: OllamaModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        prompt = "\n".join([message.content for message in model_prompt])
        print (f"Prompting: {prompt}")
        print (f"Model: {model_name}")

        response_text = await self._client.chat.create(model=model_name, prompt=prompt)
        print(f"Response 1: {response_text}")
        response_text = response_text.replace("Here is my response:", "").replace("```json", "").replace("```", "").strip()
        print(f"Response 2: {response_text}")

        assistant_message = AssistantChatMessage(content=response_text)

        try:
            assistant_reply_dict = extract_dict_from_json(assistant_message.content)
            self._logger.debug(
                "Parsing object extracted from LLM response:\n"
                f"{json.dumps(assistant_reply_dict, indent=4)}"
            )

            # 필요한 필드가 없을 경우 기본값을 설정
            if "thoughts" in assistant_reply_dict:
                if "plan" not in assistant_reply_dict["thoughts"]:
                    assistant_reply_dict["thoughts"]["plan"] = []
                if "speak" not in assistant_reply_dict["thoughts"]:
                    assistant_reply_dict["thoughts"]["speak"] = ""
                if "self_criticism" not in assistant_reply_dict["thoughts"]:
                    assistant_reply_dict["thoughts"]["self_criticism"] = ""
                if "text" not in assistant_reply_dict["thoughts"]:
                    assistant_reply_dict["thoughts"]["text"] = ""
                if "reasoning" not in assistant_reply_dict["thoughts"]:
                    assistant_reply_dict["thoughts"]["reasoning"] = ""
                if "observations" not in assistant_reply_dict["thoughts"]:
                    assistant_reply_dict["thoughts"]["observations"] = ""
            else:
                assistant_reply_dict["thoughts"] = {
                    "observations": "",
                    "text": "",
                    "reasoning": "",
                    "self_criticism": "",
                    "plan": [],
                    "speak": ""
                }

            # thoughts를 AssistantThoughts 인스턴스로 변환
            parsed_response = OneShotAgentActionProposal.parse_obj(assistant_reply_dict)
            print("parsed_response", parsed_response)
        except Exception as e:
            self._logger.error(f"Error parsing response: {e}")
            parsed_response = None

        response = ChatModelResponse(
            response=assistant_message,
            parsed_result=parsed_response,
            model_info=self.CHAT_MODELS[model_name],
            prompt_tokens_used=len(prompt),
            completion_tokens_used=len(response_text),
            completion_parser=completion_parser,
            functions=functions,
            max_output_tokens=max_output_tokens,
            prefill_response=prefill_response,
            **kwargs
        )
        print("ChatModelResponse", response.parsed_result)
        return response
    
    def get_tokenizer(self, model_name: OllamaModelName) -> ModelTokenizer[Any]:
        # HACK: No official tokenizer is available for Groq
        return tiktoken.encoding_for_model("gpt-3.5-turbo")