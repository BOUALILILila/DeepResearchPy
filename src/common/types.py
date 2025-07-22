from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Union


class AgentStopReason(StrEnum):
    TRIVIAL_ANSWER = "trivial_answer"
    MAX_BAD_ATTEMPTS = "max_bad_attempts"
    MAX_TOKENS_BUDGET = "max_tokens_budget"
    FINAL_ANSWER_OK = "final_answer_ok"


class EvaluationMetric(StrEnum):
    DEFINITIVE = "definitive"
    ATTRIBUTION = "attribution"
    FRESHNESS = "freshness"
    COMPLETENESS = "completeness"
    PLURALITY = "plurality"
    STRICT = "strict"


class KnowledgeItemType(StrEnum):
    FROM_VISIT_STEP = "from_visit_step"
    FROM_SEARCH_STEP = "from_search_step"
    FROM_ANSWER_STEP = "from_answer_step"


@dataclass
class KnowledgeItem:
    type: KnowledgeItemType
    question: str
    answer: str
    updated_at: Optional[str] = None
    references: Optional[Union[str | list[int]]] = None


class ResearchState:
    def __init__(self, user_query: str):
        self.user_query = user_query
        self.step = 1
        self.bad_attempts = 0
        self.max_bad_attempts = 2
        self.used_tokens = 0
        self.current_question = user_query
        self.gaps = [user_query]
        self.all_questions = [user_query]
        self.all_search_questions = []
        self.knowledge_items: list[KnowledgeItem] = []
        self.all_context = []
        self.all_urls: list[SearchResult] = []
        self.bad_urls = []
        self.visited_urls = []

        self.bad_actions = []
        self.steps_trace = []

        self.allow_answer = True
        self.allow_search = True
        self.allow_reflect = True
        self.allow_visit = True

        self.visited_urls = []

        self.question_evals: dict[str, list[EvaluationMetric]] = {}
        self.final_answer_pip = []
        self.stop_reason: Optional[AgentStopReason] = None


@dataclass
class SearchResult:
    url: str
    title: str
    description: str
    weight: float
