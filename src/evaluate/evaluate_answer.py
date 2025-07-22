import json
from typing import Type

from pydantic import BaseModel

from common.schemas import (
    AttributionEvaluationSchema,
    CompletenessEvaluationSchema,
    DefaultEvaluationSchema,
    PluralityEvaluationSchema,
    StrictEvaluationSchema,
)
from common.types import EvaluationMetric, KnowledgeItem
from llms.base_llm import BaseLLM
from llms.message import Message
from prompts.evaluation_prompts import (
    get_attribution_eval_prompts,
    get_completeness_eval_prompts,
    get_definitive_eval_prompts,
    get_freshness_eval_prompts,
    get_plurality_eval_prompts,
    get_strict_eval_prompts,
)
from utils.logger import get_logger

LOGGER = get_logger(__name__, step="OTHER")


class AnswerEvaluator:
    """Evaluates the agent's answer w.r.t to the defined evaluation metrics using an LLM as a judge approach"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _run_eval(self, messages: list[Message], schema: Type[BaseModel] = None) -> str:
        return self.llm.complete(messages, response_format=schema)

    def evaluate(
        self,
        question: str,
        answer: str,
        knowledge_items: list[KnowledgeItem],
        evaluation_metrics: list[EvaluationMetric],
    ) -> dict:
        """
        Evaluates the given answer against the specified evaluation metrics.

        For each evaluation metric (e.g., attribution, freshness, completeness):
        - Build the appropriate evaluation prompt
        - Invoke the LLM judge to get the evaluation
        - Returns early if any evaluation fails

        Special handling is applied for attribution:
        - If no knowledge items are available, a failure result is returned immediately.

        Parameters:
            question (str): The original user question to evaluate against.
            answer (str): The agent's generated answer.
            knowledge_items (list[KnowledgeItem]): Supporting information used for evaluations (especially attribution).
            evaluation_metrics (list[EvaluationMetric]): List of evaluation types to run on the answer.

        Returns:
            dict: A dictionary indicating evaluation results:
                - If any evaluation fails, returns a partial result with failure details.
                - If all evaluations pass, returns a success flag and summary.

        Raises:
            ValueError: If an unknown evaluation metric is provided.
        """
        results = {}
        for evaluation_type in evaluation_metrics:
            if evaluation_type == EvaluationMetric.ATTRIBUTION:
                if len(knowledge_items) == 0:
                    LOGGER.info("Knowledge items are empty for question %s", question)
                    return {
                        "pass": False,
                        "think": "The knowledge is completely empty and the answer can not be derived from it. Need to search or visit URLs.",
                        "type": "attribution",
                    }
                else:
                    schema = AttributionEvaluationSchema
                    prompts = get_attribution_eval_prompts(
                        question=question,
                        answer=answer,
                        knowledge_items=knowledge_items,
                    )
            elif evaluation_type == EvaluationMetric.DEFINITIVE:
                prompts = get_definitive_eval_prompts(question=question, answer=answer)
                schema = DefaultEvaluationSchema
            elif evaluation_type == EvaluationMetric.FRESHNESS:
                prompts = get_freshness_eval_prompts(question=question, answer=answer)
                schema = DefaultEvaluationSchema
            elif evaluation_type == EvaluationMetric.PLURALITY:
                prompts = get_plurality_eval_prompts(question=question, answer=answer)
                schema = PluralityEvaluationSchema
            elif evaluation_type == EvaluationMetric.COMPLETENESS:
                prompts = get_completeness_eval_prompts(
                    question=question, answer=answer
                )
                schema = CompletenessEvaluationSchema
            elif evaluation_type == EvaluationMetric.STRICT:
                prompts = get_strict_eval_prompts(
                    question=question, answer=answer, knowledge_items=knowledge_items
                )
                schema = StrictEvaluationSchema
            else:
                raise ValueError(f"Unknown evaluation type {evaluation_type}")

            evaluation_str = self._run_eval(messages=prompts, schema=schema)
            evaluation = json.loads(evaluation_str)
            evaluation["type"] = evaluation_type

            if not evaluation["pass"]:
                LOGGER.info(
                    "Eval %s failed for question %s: %s",
                    evaluation_type,
                    question,
                    evaluation,
                )
                return evaluation
            results["evaluation_type"] = evaluation

        LOGGER.info("All evals passed for question %s: %s", question, results)
        return {"pass": True, "think": "You passed all the tests"}
