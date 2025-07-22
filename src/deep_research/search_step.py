import json
import random
import re
import time

import tenacity
import yt_dlp
from duckduckgo_search import DDGS
from googlesearch import search as pygoogle_search

from common.deduplicate_queries import DeduplicateQueries
from common.exceptions import CouldNotSearchQuery
from common.schemas import QueryRewriteSchema
from common.types import SearchResult
from prompts.query_rewrite_prompts import get_query_rewrite_prompts
from utils.logger import get_logger
from utils.sample_k import sample_k

from .base_step import BaseStep

LOGGER = get_logger(__name__, step="SEARCH")

_DEFAULT_DIARY_MESSAGE = """
At step {step}, you took the **search** action and look for external information for the question: "{current_question}".
In particular, you tried to search for the following keywords: "{formatted_keyword_queries}".
You found quite some information and add them to your URL list and **visit** them later when needed. 
"""

_NO_RESULT_DIARY_MESSAGE = """
At step {step}, you took the **search** action and look for external information for the question: "{current_question}".
In particular, you tried to search for the following keywords: {formatted_keyword_queries}. 
But then you realized you have already searched for these keywords before, no new information is returned.
You decided to think out of the box or cut from a completely different angle.
"""


class SearchStep(BaseStep):
    """
    Handles a search action.
    Re-writes, deduplicates and searches the queries on the web using google and duckduckgo, and records the trace in the agent's diary.
    """

    def __init__(
        self,
        queries,
        state,
        question_deduplicator,
        llm,
        action_think: str,
        max_requests: int = 5,
        max_search_results: int = 5,
    ) -> None:
        super().__init__(state)
        self.queries = queries
        self.max_requests = max_requests
        self.action_think = action_think
        self.llm = llm
        self.max_search_results = max_search_results
        self.question_deduplicator: DeduplicateQueries = question_deduplicator

    def __repr__(self):
        return f"SearchStep(step={self.state.step}, queries={self.queries}, max_requests={self.max_search_results}, max_search_results={self.max_search_results})"

    def as_markdown(self):
        f_queries = "\n- ".join(self.queries)
        return f"""Searching for:\n- {f_queries}"""

    def deduplicate_questions(
        self,
        all_questions: list[str],
        current_questions: list[str],
        k: int = 5,
    ):
        dedup_queries = self.question_deduplicator.dedup(
            current_questions, all_questions
        )
        return sample_k(dedup_queries, k)

    def rewrite_queries(
        self, queries: list[str], initial_search_results: str
    ) -> list[str]:
        rewritten_queries = []
        for query in queries:
            LOGGER.info("Rewriting the query: %s", query)
            rewrite_prompt_messages = get_query_rewrite_prompts(
                query=query,
                think=self.action_think,
                initial_search_results=initial_search_results,
            )
            response = self.llm.complete(
                messages=rewrite_prompt_messages, response_format=QueryRewriteSchema
            )

            LOGGER.info("Into %s", response)

            rewritten_queries.extend(json.loads(response)["queries"])
        return rewritten_queries

    @tenacity.retry(
        wait=tenacity.wait_fixed(4),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(CouldNotSearchQuery),
        reraise=True,
    )
    def google_search(self, search_query) -> list[SearchResult]:
        LOGGER.info("(Google) Searching for query: %s", search_query)
        try:
            results = pygoogle_search(
                search_query,
                num_results=self.max_search_results,
                safe=None,
                unique=True,
                advanced=True,
            )
            return [
                self.process_search_result(
                    url=result.url,
                    title=result.title.strip(),
                    description=result.description.strip(),
                    weight=1,
                )
                for result in results
            ]
        except Exception:
            raise CouldNotSearchQuery(search_query)

    @tenacity.retry(
        wait=tenacity.wait_fixed(4),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(CouldNotSearchQuery),
        reraise=True,
    )
    def duckduck_go_search(self, search_query) -> list[SearchResult]:
        try:
            results = DDGS().text(search_query, max_results=self.max_search_results)
            time.sleep(2)  # avoid throttling
            return [
                self.process_search_result(
                    url=result["href"],
                    title=result["title"].strip(),
                    description=result["body"].strip(),
                    weight=1,
                )
                for result in results
            ]
        except Exception:
            raise CouldNotSearchQuery(search_query)

    def execute_search_queries(self, search_queries):
        successfully_searched_queries = []
        new_knowledge_items = []
        for query in search_queries:
            try:
                if random.random() < 0.5:
                    search_results = self.duckduck_go_search(search_query=query)
                else:
                    search_results = self.google_search(search_query=query)
            except CouldNotSearchQuery:
                continue

            # knowledge_item = KnowledgeItem(
            #    question=f'What do internet say about "{query}"?',
            #    answer="; ".join([res.description for res in search_results]),
            #    type=KnowledgeItemType.FROM_SEARCH_STEP,
            #    updated_at=get_current_datetime(),
            # )
            # new_knowledge_items.append(knowledge_item)

            self.state.all_urls.extend(search_results)

            successfully_searched_queries.append(query)

        return new_knowledge_items, successfully_searched_queries

    def normalize_arxiv_url(self, url: str) -> str:
        match = re.match(r"https?://arxiv\.org/pdf/(\d+\.\d+)(v\d+)?\.pdf", url)
        if match:
            # Use the html version of the article for readability
            return f"https://arxiv.org/html/{match.group(1)}"
        return url

    def enrich_youtube_metadata(
        self, url: str, title: str, description: str
    ) -> tuple[str, str]:
        if "youtube.com/watch" in url or "youtu.be/" in url:
            try:
                with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return info.get("title", title), info.get(
                        "description", description
                    )
            except Exception:
                pass
        return title, description

    def process_search_result(
        self, url: str, title: str, description: str, weight: float = 1
    ) -> SearchResult:
        url = self.normalize_arxiv_url(url)
        title, description = self.enrich_youtube_metadata(url, title, description)
        return SearchResult(
            url=url, title=title, description=description, weight=weight
        )

    def handle(self):
        LOGGER.info("Handling %s", self)

        # Remove semantically similar queries and sample k queries max to search
        self.queries = self.deduplicate_questions(
            all_questions=[],
            current_questions=self.queries,
            k=self.max_requests,
        )

        # Perform an initial web search
        new_knowledge_items, successfully_search_queries = self.execute_search_queries(
            self.queries
        )
        self.state.knowledge_items.extend(new_knowledge_items)
        self.state.all_search_questions.extend(successfully_search_queries)

        # Construct the initial search context from the results of the web search
        initial_search_results = [item.answer for item in new_knowledge_items]

        # Re-write and expand queries using the initial search context feedback
        rewritten_keyword_queries = self.rewrite_queries(
            queries=self.queries, initial_search_results=initial_search_results
        )

        # Search the expanded queries
        rewritten_queries_knowledge_items, searched_rewritten_queries = (
            self.execute_search_queries(
                search_queries=[query["q"] for query in rewritten_keyword_queries]
            )
        )

        if len(searched_rewritten_queries) > 0:
            self.state.knowledge_items.extend(rewritten_queries_knowledge_items)
            self.state.all_search_questions.extend(searched_rewritten_queries)

            self.state.steps_trace.append(
                _DEFAULT_DIARY_MESSAGE.format(
                    step=self.state.step,
                    current_question=self.state.current_question,
                    formatted_keyword_queries=", ".join(
                        [query["q"] for query in rewritten_keyword_queries]
                    ),
                )
            )

        if len(searched_rewritten_queries) == 0 or len(rewritten_keyword_queries) == 0:
            self.state.steps_trace.append(
                _NO_RESULT_DIARY_MESSAGE.format(
                    step=self.state.stpe,
                    current_question=self.state.current_question,
                    formatted_keyword_queries=", ".join(
                        [query["q"] for query in rewritten_keyword_queries]
                    ),
                )
            )

        self.state.allow_search = False
