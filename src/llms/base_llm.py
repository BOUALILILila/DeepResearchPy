from abc import ABC, abstractmethod

from pydantic import BaseModel

from .message import Message


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = None,
        response_format: type[BaseModel] = None,
    ) -> str:
        raise NotImplementedError
