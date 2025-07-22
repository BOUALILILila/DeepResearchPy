import os
from enum import Enum

from llms.mistral import MistralLLM
from llms.openai import OpenAILLM

from .base_llm import BaseLLM


class Provider(str, Enum):
    OPENAI = "openai"
    MISTRAL = "mistral"


def get_model(provider: Provider, model_name: str) -> BaseLLM:
    if provider == Provider.MISTRAL:
        if "MISTRAL_API_KEY" not in os.environ:
            raise ValueError("Could not find env variable 'MISTRAL_API_KEY'")
        return MistralLLM(api_key=os.getenv("MISTRAL_API_KEY"), model_name=model_name)
    if provider == Provider.OPENAI:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Could not find env variable 'OPENAI_API_KEY'")
        return OpenAILLM(api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name)
    else:
        raise ValueError(f"Unsupported provider '{provider}'")
