import json
from dataclasses import asdict

import tenacity
from openai import OpenAI
from pydantic import BaseModel

from .base_llm import BaseLLM
from .message import Message


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4.1",
    ):
        super().__init__(model_name=model_name)
        self._client = OpenAI(api_key=api_key)
        self._used_tokens = 0

    def convert_messages(self, messages: list[Message]) -> list[dict]:
        return [self.transform_message(message) for message in messages]

    def transform_message(self, message: Message) -> dict:
        msg = asdict(message)
        msg["role"] = "developer" if msg["role"] == "system" else msg["role"]
        return msg

    @property
    def used_tokens(self):
        return self._used_tokens

    @tenacity.retry(
        wait=tenacity.wait_fixed(1),
        stop=tenacity.stop_after_attempt(2),
        retry=tenacity.retry_if_exception_type(json.decoder.JSONDecodeError),
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
            chat_response = self._client.responses.parse(
                model=self.model_name,
                input=self.convert_messages(messages),
                temperature=temperature,
                max_output_tokens=max_tokens,
                text_format=response_format,
            )
        else:
            chat_response = self._client.responses.create(
                model=self.model_name,
                input=self.convert_messages(messages),
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        self._used_tokens += chat_response.usage.total_tokens
        return chat_response.choices[0].message.content
