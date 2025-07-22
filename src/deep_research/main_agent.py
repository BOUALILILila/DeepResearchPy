import json
from typing import Union

from dotenv import load_dotenv
from pydantic import Field, create_model

from common.cherry_picker import CherryPicker
from common.config import Configuration
from common.deduplicate_queries import DeduplicateQueries
from common.schemas import (
    AnswerAction,
    AnswerActionContent,
    ReflectAction,
    SearchAction,
    VisitAction,
)
from common.semantic_similarity import SemanticSimilarityScorer
from common.types import (
    AgentStopReason,
    EvaluationMetric,
    KnowledgeItem,
    ResearchState,
    SearchResult,
)
from evaluate.evaluate_answer import AnswerEvaluator
from evaluate.evaluate_question import QuestionEvaluator
from llms import get_model
from llms.message import Message
from prompts.main_agent_prompts import get_main_agent_prompt
from utils.logger import get_logger

from .answer_step import AnswerStep
from .base_step import BaseStep
from .reflect_step import ReflectStep
from .search_step import SearchStep
from .visit_step import VisitStep

load_dotenv()

LOGGER = get_logger(__name__, step="SEARCH")


class DeepResearch:
    def __init__(self, config: Configuration):
        self.state = None
        self.config = config

        self.llm = get_model(
            provider=config.model_provider, model_name=config.model_name
        )

        self.answer_evaluator = AnswerEvaluator(llm=self.llm)
        self.question_evaluator = QuestionEvaluator(llm=self.llm)
        self.question_deduplicator = DeduplicateQueries(llm=self.llm)
        self.semantic_similarity_scorer = SemanticSimilarityScorer(
            batch_size=config.semantic_similarity.batch_size,
            max_length=config.semantic_similarity.max_length,
        )
        self.cherry_picker = CherryPicker(
            similarity_scorer=self.semantic_similarity_scorer,
            chunk_size=config.snippet_extraction.chunk_size,
            n_snippets=config.snippet_extraction.num_snippets,
            snippets_length=config.snippet_extraction.snippet_length,
        )

    def get_prompt(
        self,
        knowledge_items: list[KnowledgeItem],
        action_history: list,
        bad_actions: list,
        urls_to_visit: list[SearchResult],
        enforce_answer: bool = False,
    ):
        available_actions = []
        if not enforce_answer:
            if self.state.allow_search:
                available_actions.append("search")
            if self.state.allow_answer:
                available_actions.append("answer")
            if self.state.allow_reflect:
                available_actions.append("reflect")
            if self.state.allow_visit:
                available_actions.append("visit")

        assert (len(available_actions) > 0 and not enforce_answer) or (
            enforce_answer and len(available_actions) == 0
        ), "Check enforce_answer and available_actions"

        return get_main_agent_prompt(
            knowledge_items=knowledge_items,
            action_history=action_history,
            bad_actions=bad_actions,
            available_actions=available_actions,
            urls_to_visit=urls_to_visit,
            enforce_answer=enforce_answer,
            max_search_queries=self.config.search_step.max_questions_to_search,
            max_decomposition_questions=self.config.reflect_step.max_decomposition_questions,
        )

    def parse_current_step(self, response: str) -> BaseStep:
        response = json.loads(response)
        action_name = list(response["action"].keys())[0]
        action_content = response["action"][action_name]
        if action_name == "search":
            return SearchStep(
                queries=action_content["queries"],
                state=self.state,
                question_deduplicator=self.question_deduplicator,
                llm=self.llm,
                action_think=response["think"],
                max_requests=self.config.search_step.max_questions_to_search,
                max_search_results=self.config.search_step.top_k_search_results,
            )
        if action_name == "answer":
            return AnswerStep(
                answer=action_content["answer"],
                references=(
                    action_content["references"]
                    if "references" in action_content
                    else []
                ),
                state=self.state,
                answer_evaluator=self.answer_evaluator,
                llm=self.llm,
                max_bad_attempts=self.config.answer_step.max_bad_attempts,
            )
        if action_name == "reflect":
            return ReflectStep(
                questions_to_answer=action_content["questions_to_answer"],
                state=self.state,
                question_deduplicator=self.question_deduplicator,
                max_questions_to_answer=self.config.reflect_step.max_decomposition_questions,
            )
        if action_name == "visit":
            return VisitStep(
                urls=action_content["urls"],
                state=self.state,
                cherry_picker=self.cherry_picker,
                max_urls_per_step=self.config.visit_step.max_urls_to_visit,
            )
        if action_name == "code":
            raise NotImplementedError("Coming soon...")

        raise NotImplementedError(f"Unknown action name: {action_name}")

    def evaluate_question(self, question: str) -> list[EvaluationMetric]:
        return self.question_evaluator.evaluate(
            question=question,
        )

    def rerank_urls(self, urls: list[SearchResult]) -> list[SearchResult]:
        if len(urls) == 0:
            return []
        # Construct the urls descriptor strings
        url_descriptors = []
        for result in urls:
            output = result.url + ": " + result.title
            if len(result.description):
                output += f" - {result.description}"
            url_descriptors.append(output)

        # Score the url descriptors w.r.t to the current question
        scores = self.semantic_similarity_scorer.compute_similarities(
            query=self.state.current_question, docs=url_descriptors
        )

        # Sort the urls w.r.t to their scores
        sorted_indices = [
            i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ]
        reranked_urls = []
        for idx in sorted_indices[: self.config.top_k_urls_rerank]:
            result = urls[idx]
            score = scores[idx]
            result.weight = score
            reranked_urls.append(result)

        return reranked_urls

    def get_user_msg(self, final_answer_pip: list[str] = None) -> str:
        user_msg = f"<question> {self.state.current_question} </question>"

        if final_answer_pip and len(final_answer_pip) > 0:
            reviews = "\n".join(
                [
                    f"<reviewer-{idx}>{p}</reviewer-{idx}>"
                    for idx, p in enumerate(final_answer_pip, start=1)
                ]
            )

            user_msg += f"""
<answer-requirements>
- You provide deep, unexpected insights, identifying hidden patterns and connections, and creating "aha moments.".
- You break conventional thinking, establish unique cross-disciplinary connections, and bring new perspectives to the user.
- Follow reviewer's feedback and improve your answer quality:
{reviews}
</answer-requirements>"""

        return user_msg

    def get_output_schema(self):
        available_action_schemas = []
        if self.state.allow_answer:
            available_action_schemas.append(AnswerAction)

        if self.state.allow_search:
            available_action_schemas.append(SearchAction)

        if self.state.allow_visit:
            available_action_schemas.append(VisitAction)

        if self.state.allow_reflect:
            available_action_schemas.append(ReflectAction)

        return create_model(
            "StepOutputSchema",
            think=(
                str,
                Field(
                    description="Concisely explain your reasoning process",
                ),
            ),
            action=(
                Union[tuple(available_action_schemas)],
                Field(
                    description="Choose exactly one action from the available actions, fill in the corresponding action schema."
                ),
            ),
        )

    def get_final_answer(self):
        self.state.current_question = self.state.user_query

        enforce_answer_sys_prompt = self.get_prompt(
            action_history=self.state.steps_trace,
            bad_actions=self.state.bad_actions,
            knowledge_items=self.state.knowledge_items,
            urls_to_visit=self.state.all_urls,
            enforce_answer=True,
        )

        final_answer_output_schema = create_model(
            "Answer",
            think=(
                str,
                Field(
                    description="Concisely explain your reasoning process",
                ),
            ),
            answer=(
                AnswerActionContent,
                Field(description="A detailed definitive answer."),
            ),
        )

        # invoke LLM prediction on current question
        response = self.llm.complete(
            messages=[
                Message(role="system", content=enforce_answer_sys_prompt),
                Message(
                    role="user",
                    content=self.get_user_msg(self.state.final_answer_pip),
                ),
            ],
            response_format=final_answer_output_schema,
        )

        response = json.loads(response)
        action = response["answer"]
        current_step = AnswerStep(
            answer=action["answer"],
            references=action["references"],
            state=self.state,
            answer_evaluator=self.answer_evaluator,
            llm=self.llm,
            eval_answer=False,
        )
        current_step.handle()
        return current_step

    def __call__(self, user_query: str):
        self.state = ResearchState(user_query=user_query)

        # Leave out a proportion of the allowed budget for writing a final answer
        real_budget = self.config.max_token_budget * 0.85

        while self.llm.used_tokens < real_budget:
            self.state.current_question = (
                self.state.user_query
                if len(self.state.gaps) == 0
                else self.state.gaps.pop()
            )

            if (
                self.state.current_question == self.state.user_query
                and self.state.step == 1
            ):
                # only add evaluation for initial question, once at step 1
                self.state.question_evals[self.state.current_question] = (
                    self.evaluate_question(question=self.state.current_question)
                )
                # force strict eval for the original question, only once.
                self.state.question_evals[self.state.current_question].append(
                    EvaluationMetric.STRICT
                )
            elif self.state.current_question != self.state.user_query:
                self.state.question_evals[self.state.current_question] = []

            if (
                self.state.step == 1
                and "freshness"
                in self.state.question_evals[self.state.current_question]
            ):
                # if it detects freshness, avoid direct answer at step 1
                self.state.allow_answer = False
                self.state.allow_reflect = False

            # rerank URLs
            top_rearanked_urls = self.rerank_urls(urls=self.state.all_urls)

            # Get the step prompt
            current_sys_prompt = self.get_prompt(
                action_history=self.state.steps_trace,
                bad_actions=self.state.bad_actions,
                knowledge_items=self.state.knowledge_items,
                urls_to_visit=top_rearanked_urls,
            )

            output_schema = self.get_output_schema()

            # invoke LLM prediction on current question
            current_step_response = self.llm.complete(
                messages=[
                    Message(role="system", content=current_sys_prompt),
                    Message(
                        role="user",
                        content=self.get_user_msg(
                            self.state.final_answer_pip
                            if self.state.current_question == user_query
                            else None
                        ),
                    ),
                ],
                response_format=output_schema,
            )

            current_step = self.parse_current_step(response=current_step_response)

            # reset allows to true
            self.state.allow_answer = True
            self.state.allow_search = True
            self.state.allow_reflect = True
            self.state.allow_visit = True

            current_step.handle()

            if self.state.stop_reason:
                LOGGER.info("Stop Reason: %s", self.state.stop_reason)
                if self.state.stop_reason in [
                    AgentStopReason.TRIVIAL_ANSWER,
                    AgentStopReason.FINAL_ANSWER_OK,
                ]:
                    LOGGER.info("Here is your answer:\n %s", current_step.answer)

                    yield current_step, True
                    return

                if self.state.stop_reason == AgentStopReason.MAX_BAD_ATTEMPTS:
                    LOGGER.info(
                        "Maximum bad attempts reached, trying to get a final answer nevertheless."
                    )
                    break
            else:
                yield current_step, False

            if self.llm.used_tokens >= real_budget:
                self.state.stop_reason = AgentStopReason.MAX_TOKENS_BUDGET
                break

            self.state.step += 1

        if self.state.stop_reason in [
            AgentStopReason.MAX_TOKENS_BUDGET,
            AgentStopReason.MAX_BAD_ATTEMPTS,
        ]:
            LOGGER.info("STOPPED because: %s", self.state.stop_reason)
            LOGGER.info("Enforcing Answer...")

            # Try and get a final answer, better than nothing
            current_step = self.get_final_answer()
            yield current_step, True

        else:
            raise ValueError("Something unexpected happened... please try again")
