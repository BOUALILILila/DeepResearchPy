import json

from common.schemas import QuestionEvaluationSchema
from common.types import EvaluationMetric
from llms.base_llm import BaseLLM
from prompts.evaluation_prompts import get_question_eval_prompts


class QuestionEvaluator:
    """Specifies the evaluation metrics required for assessing the agent's answer."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def evaluate(
        self,
        question: str,
    ) -> list[EvaluationMetric]:
        response = self.llm.complete(
            messages=get_question_eval_prompts(question=question),
            response_format=QuestionEvaluationSchema,
        )

        evaluation_metrics = []
        evaluation_output = json.loads(response)
        if (
            "needs_definitive" in evaluation_output
            and evaluation_output["needs_definitive"]
        ):
            evaluation_metrics.append(EvaluationMetric.DEFINITIVE)
        if (
            "needs_freshness" in evaluation_output
            and evaluation_output["needs_freshness"]
        ):
            evaluation_metrics.append(EvaluationMetric.FRESHNESS)
        if (
            "needs_completeness" in evaluation_output
            and evaluation_output["needs_completeness"]
        ):
            evaluation_metrics.append(EvaluationMetric.COMPLETENESS)
        if (
            "needs_plurality" in evaluation_output
            and evaluation_output["needs_plurality"]
        ):
            evaluation_metrics.append(EvaluationMetric.COMPLETENESS)
        return evaluation_metrics
