import json

from common.schemas import DeduplicateQueriesSchema
from llms.base_llm import BaseLLM
from prompts.deduplicate_prompts import get_query_dedup_prompts


class DeduplicateQueries:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def dedup(
        self,
        new_queries: list[str],
        queries: list[str],
    ) -> list[str]:
        queries = queries + new_queries
        dedup_queries_output = self.llm.complete(
            messages=get_query_dedup_prompts(queries=queries),
            response_format=DeduplicateQueriesSchema,
        )

        return json.loads(dedup_queries_output)["queries"]
