import logging
from typing import List, Optional, Any, Mapping

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.trubrics_callback import _convert_message_to_dict
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage

import erniebot

from pandasai.helpers import Logger
from pydantic import BaseModel

DEFAULT_MODEL_NAME = "ernie-bot"

logger = Logger()


class LLMTokenUsage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0


class AIStudioErnieBot(BaseChatModel):
    """
    Baidu AI Studio mode, support ernie-bot and ernie-bot-turbo.
    WARNING: aistudio mode is not compatible with multi tenants.
    """
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = 0.1
    access_token: Optional[str] = None

    def __init__(self, access_token: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.access_token = access_token
        logger.log(f"Baidu aistudio is used.", level=logging.INFO)
        erniebot.api_type = "aistudio"
        if erniebot.access_token is None or erniebot.access_token != access_token:
            erniebot.access_token = access_token
        self.model_name = kwargs.get("model_key", DEFAULT_MODEL_NAME)

    @property
    def _llm_type(self) -> str:
        return "baidu-as-ernie-bot"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        logger.log(f"Payload for ernie api is {messages}", level=logging.DEBUG)
        resp = erniebot.ChatCompletion.create(
            model=self.model_name,
            messages=[_convert_message_to_dict(m) for m in messages],
            temperature=self.temperature,
            top_p=0.95,
            stream=False,
        )
        if resp.get("error_code"):
            raise ValueError(f"Error from BaiduAIStudioErnieBot api response: {resp}")

        return self._create_chat_result(resp)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = [
            ChatGeneration(message=AIMessage(content=response.get("result")))
        ]
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        total_usage = LLMTokenUsage()
        for output in llm_outputs:
            if output is None:
                continue

            usage = output["token_usage"]
            total_usage.total_tokens += int(usage["total_tokens"])
            total_usage.completion_tokens += int(usage["completion_tokens"])
            total_usage.prompt_tokens += int(usage["prompt_tokens"])

        return total_usage.dict()
