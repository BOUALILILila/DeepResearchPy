from typing import Annotated

from pydantic import BaseModel, Field


class DeduplicateQueriesSchema(BaseModel):
    queries: list[str] = Field(description="List of deduplicated queries")


# ------------------------------- Action Schemas -----------------------------------
class AnswerActionContent(BaseModel):
    references: list[int] = Field(
        description=(
            "A list of integer indices corresponding to the relevant knowledge items "
            "from the <knowledge> section that were used to support the answer. "
            "Each index should uniquely identify a cited knowledge item."
        ),
    )
    answer: str = Field(
        description=(
            "A definitive, unambiguous, and well-supported response using the relevant knowledge items. "
            "The answer must be written in Markdown format and include inline citations using square brackets "
            "to refer to the knowledge item indices, e.g., [2] for citing knowledge item #2. "
            "Avoid speculative or uncertain language; provide a clear and authoritative answer based on the evidence."
        )
    )


class AnswerAction(BaseModel):
    answer: AnswerActionContent = Field(description="An answer action.")


# -----

Question = Annotated[
    str,
    Field(
        description="A single line question. It must be: Original (not variations of existing questions); Focused on single concepts; Under 20 words; Non-compound/non-complex"
    ),
]


class ReflectActionContent(BaseModel):
    questions_to_answer: list[Question] = Field(
        description="The list of most important questions to fill the knowledge gaps of finding the answer to the original question.",
    )


class ReflectAction(BaseModel):
    reflect: ReflectActionContent = Field(description="A reflect action")


# -----

SearchQuery = Annotated[
    str,
    Field(
        description="A natural language search request. Based on the deep intention behind the original question and the expected answer format."
    ),
]


class SearchActionContent(BaseModel):
    queries: list[SearchQuery] = Field(
        description="The list of queries. Always prefer a single query, only add another query if the original question covers multiple aspects or elements and one search query is definitely not enough, each query focuses on one specific aspect of the original question. Minimize mutual information between all queries.",
    )


class SearchAction(BaseModel):
    search: SearchActionContent = Field(description="A search action")


# -----


class VisitActionContent(BaseModel):
    urls: list[str] = Field(
        description="The list of URLs to visit, choose up to 5 relevant URLs from the list of available URLs.",
    )


class VisitAction(BaseModel):
    visit: VisitActionContent = Field(description="A visit action.")


# ----------------------------------------------------------------------


# ------------------------- Evaluation Schemas -------------------------
class ErrorAnalysisSchema(BaseModel):
    recap: str = Field(description="Recap key actions and highlight what went wrong")
    blame: str = Field(
        description="Point to specific steps or patterns that led to the inadequate answer"
    )
    imporvement: str = Field(
        description="Actionable suggestions to get a better outcome"
    )


class QuestionEvaluationSchema(BaseModel):
    think: str = Field(
        description="A concise explanation of why these checks are needed."
    )
    needs_definitive: bool
    needs_freshness: bool
    needs_plurality: bool
    needs_completeness: bool


class DefaultEvaluationSchema(BaseModel):
    think: str = Field(
        description="Explain the thought process why the answer pass or does not pass the evaluation"
    )
    pass_: bool = Field(
        description="Boolean value (true or false). True if the answer passes the test, false otherwise",
        alias="pass",
    )


class AttributionEvaluationSchema(BaseModel):
    think: str = Field(
        description="Explain the thought process why the answer pass or does not pass the evaluation"
    )
    pass_: bool = Field(
        description="Boolean value (true or false). True if the answer passes the test, false otherwise",
        alias="pass",
    )

    exact_quote: str = Field(
        description="Exact relevant quote and evidence from the source that strongly support the answer and justify this question-answer pair, if any"
    )


class CompletenessEvaluationSchema(BaseModel):
    think: str = Field(
        description="Explain the thought process why the answer pass or does not pass the evaluation"
    )
    pass_: bool = Field(
        description="Boolean value (true or false). True if the answer passes the test, false otherwise",
        alias="pass",
    )
    aspects_expected: str = Field(
        description="Comma-separated list of all aspects or dimensions that the question explicitly asks for"
    )
    aspects_provided: str = Field(
        description="Comma-separated list of all aspects or dimensions that were actually addressed in the answer"
    )


class PluralityEvaluationSchema(BaseModel):
    think: str = Field(
        description="Explain the thought process why the answer pass or does not pass the evaluation"
    )
    pass_: bool = Field(
        description="Boolean value (true or false). True if the answer passes the test, false otherwise",
        alias="pass",
    )
    min_count_required: int = Field(
        description="Minimum required number of items from the **question**"
    )
    actual_count_provided: int = Field(
        description="Number of items provided in **answer**"
    )


class StrictEvaluationSchema(BaseModel):
    think: str = Field(
        description="Explain the thought process why the answer pass or does not pass the evaluation"
    )
    pass_: bool = Field(
        description="Boolean value (true or false). True if the answer passes the test, false otherwise",
        alias="pass",
    )
    improvement_plan: str = Field(
        description='Explain how an ideal answer should look like and propose improvements for the current answer. Start with "For the best answer, you must..."'
    )


# ----------------------------------------------------------------------------


# ------------------------- Query Re-writing Schemas -------------------------
class QuerySchema(BaseModel):
    q: str = Field(
        description="Keyword-based web search query, 1-3 words preferred, total length < 30 characters"
    )


class QueryRewriteSchema(BaseModel):
    think: str = Field(
        description="**Concisely** explain why you choose those search queries"
    )
    queries: list[QuerySchema] = Field(
        description="Array of search keywords queries, orthogonal to each other."
    )


# ----------------------------------------------------------------------------
