from typing import Optional

import yaml
from pydantic import BaseModel, Field

from llms import Provider


class ReflectStepConfig(BaseModel):
    max_decomposition_questions: Optional[int] = Field(
        default=3,
        description="Maximum number of sub questions to create from the original question.",
    )


class SearchStepConfig(BaseModel):
    max_questions_to_search: Optional[int] = Field(
        default=3,
        description="Maximum number of unique sub questions to search.",
    )
    top_k_search_results: Optional[int] = Field(
        default=5,
        description="Top k search results for each question.",
    )


class VisitStepConfig(BaseModel):
    max_urls_to_visit: Optional[int] = Field(
        default=5,
        description="Maximum number of urls to visit.",
    )


class AnswerStepConfig(BaseModel):
    max_bad_attempts: Optional[int] = Field(
        default=2,
        description="Maximum number of failed answer generation attempts before aborting.",
    )


class SnippetExtractionConfig(BaseModel):
    chunk_size: int = Field(
        default=100,
        description="Size (in characters) of each document chunk to be processed.",
    )
    num_snippets: int = Field(
        default=10,
        description="Maximum number of top relevant document snippets to include in the context.",
    )
    snippet_length: int = Field(
        default=400, description="Length (in characters) of each extracted snippet."
    )
    min_similarity: float = Field(
        default=0.3,
        description="Minimum cosine similarity threshold required for a snippet to be included.",
    )


class SemanticSimilarityConfig(BaseModel):
    batch_size: int = Field(default=32, description="")
    max_length: int = Field(default=512, description="")


class Configuration(BaseModel):
    """Configuration settings for the deep research agent, loaded from a YAML file."""

    model_provider: Provider = Field(
        description="The name of the provider to use in (mistral, openai)"
    )
    model_name: str = Field(
        description="Name or identifier of the language model to use (e.g., 'mistral-medium-latest', 'gpt-4o').",
    )
    max_token_budget: Optional[int] = Field(
        default=50_000,
        description="Upper limit on the total number of tokens allowed for a full response context.",
    )
    reflect_step: Optional[ReflectStepConfig] = Field(
        default_factory=ReflectStepConfig,
        description="Configuration options for the Reflect Step.",
    )
    search_step: Optional[SearchStepConfig] = Field(
        default_factory=SearchStepConfig,
        description="Configuration options for the Search Step.",
    )
    visit_step: Optional[VisitStepConfig] = Field(
        default_factory=VisitStepConfig,
        description="Configuration options for the Visit Step.",
    )
    answer_step: Optional[AnswerStepConfig] = Field(
        default_factory=AnswerStepConfig,
        description="Configuration options for the Answer Step.",
    )
    top_k_urls_rerank: Optional[int] = Field(
        default=20,
        description="Top k relevant urls to include in the agent's context for the current question after reranking.",
    )
    snippet_extraction: Optional[SnippetExtractionConfig] = Field(
        default_factory=SnippetExtractionConfig,
        description="Configuration options for snippet extraction and filtering.",
    )
    semantic_similarity: Optional[SemanticSimilarityConfig] = Field(
        default_factory=SemanticSimilarityConfig,
        description="Configuration options for the semantic similarity estimation.",
    )

    @classmethod
    def from_yaml(cls, path: str) -> "Configuration":
        """
        Loads and validates the configuration from a YAML file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Configuration: A validated configuration object.

        Raises:
            ValidationError: If the YAML content is invalid or missing required fields.
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the file cannot be parsed as valid YAML.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
