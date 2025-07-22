import json
from dataclasses import asdict

import mistralai
import tenacity
from mistralai import Mistral
from pydantic import BaseModel

from .base_llm import BaseLLM
from .message import Message


class MistralLLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model_name: str = "mistral-large-latest",
        seed: int = 1234,
    ):
        super().__init__(model_name=model_name)
        self._client = Mistral(api_key=api_key)
        self._used_tokens = 0
        self.seed = seed

    def convert_messages(self, messages: list[Message]) -> list[dict]:
        return [asdict(message) for message in messages]

    @property
    def used_tokens(self):
        return self._used_tokens

    @tenacity.retry(
        wait=tenacity.wait_fixed(5),
        stop=tenacity.stop_after_attempt(2),
        retry=tenacity.retry_if_exception_type(
            (json.decoder.JSONDecodeError, mistralai.models.sdkerror.SDKError)
        ),
        reraise=True,
    )
    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = None,
        response_format: type[BaseModel] = None,
    ) -> str:
        if response_format:
            chat_response = self._client.chat.parse(
                model=self.model_name,
                messages=self.convert_messages(messages),
                random_seed=self.seed,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        else:
            chat_response = self._client.chat.complete(
                model=self.model_name,
                messages=self.convert_messages(messages),
                random_seed=self.seed,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        self._used_tokens += chat_response.usage.total_tokens
        return chat_response.choices[0].message.content
