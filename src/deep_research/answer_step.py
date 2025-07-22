import json

from common.schemas import ErrorAnalysisSchema
from common.types import AgentStopReason, KnowledgeItem, KnowledgeItemType
from evaluate.evaluate_answer import AnswerEvaluator
from llms.base_llm import BaseLLM
from prompts.error_analysis_prompts import get_analyze_step_prompts
from utils.date_utils import get_current_datetime
from utils.logger import get_logger

from .base_step import BaseStep

LOGGER = get_logger(__name__, step="ANSWER")


def get_error_analysis(llm, trace: list[str]) -> dict:
    output = llm.complete(
        messages=get_analyze_step_prompts(steps_trace=trace),
        response_format=ErrorAnalysisSchema,
    )

    analysis = {
        "recap": "",
        "blame": "",
        "improvement": "",
    }
    analysis.update(json.loads(output))
    return analysis


_MAIN_QUESTION_PASS_EVAL_DIARY = """
At step {step}, you took **answer** action and finally found the answer to the original question:

Original question: 
{current_question}

Your answer:
{answer}

The evaluator thinks your answer is good because:
{eval_current_answer_think}

You have successfully answered the original question"""

_MAIN_QUESTION_FAIL_EVAL_DIARY = """
At step {step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
{current_question}

Your answer: 
{answer}

The evaluator thinks your answer is bad because: 
{eval_current_answer_think}
"""

_SUB_QUESTION_PASS_EVAL_DIARY = """
At step {step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
{current_question}

Your answer: 
{answer}

The evaluator thinks your answer is good because: 
{eval_current_answer_think}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.
"""


class AnswerStep(BaseStep):
    """
    Handles an answer action.
    Evaluates the agent's generated answer against the evaluation metrics using an LLM as a judge, and records the answer step with its evaluation in the agent's diary for traceability.
    """

    def __init__(
        self,
        answer: str,
        references: list[int],
        llm: BaseLLM,
        state,
        answer_evaluator: AnswerEvaluator,
        max_bad_attempts: int = 2,
        eval_answer: bool = True,
    ):
        super().__init__(state=state)
        self.max_bad_attempts = max_bad_attempts
        self.answer = answer
        self.references = references
        self.answer_evaluator = answer_evaluator
        self.llm = llm
        self.eval_answer = eval_answer
        self.bad_attempt = False

    def __repr__(self):
        return f"AnswerStep(step={self.state.step}, answer={self.answer}, references={self.references}, max_bad_attempts={self.max_bad_attempts}, current_question={self.state.current_question})"

    def as_markdown(self):
        if self.state.current_question == self.state.user_query:
            output = "Generating a final answer for the user question..."
            if self.bad_attempt:
                return f"{output}\n\nThe generated anwser was rejected since it failed to satsify the evaluation criteria."
        return f"Answering question: {self.state.current_question}"

    def evaluate_answer(
        self, question: str, answer: str, knowledge_items: list[KnowledgeItem]
    ) -> dict:
        return self.answer_evaluator.evaluate(
            question=question,
            answer=answer,
            knowledge_items=knowledge_items,
            evaluation_metrics=self.state.question_evals[question],
        )

    def clean_references(self, references: list[int]) -> list[int]:
        """
        Filters and returns a list of valid knowledge item indices.

        This method verifies that each index in the provided `references` list refers to a
        valid knowledge item within the current knowledge base (`self.state.knowledge_items`).
        Indices that are out of bounds (i.e., greater than or equal to the number of available
        knowledge items) are excluded from the returned list.

        Parameters:
            references (list[int]): A list of knowledge item indices to validate.

        Returns:
            list[int]: A list containing only the valid knowledge item indices.
        """

        knowledge_size = len(self.state.knowledge_items)
        curated_knowledge_references = [
            ref for ref in references if 0 <= ref < knowledge_size
        ]
        return curated_knowledge_references

    def analyze_steps(self):
        return get_error_analysis(llm=self.llm, trace=self.state.steps_trace)

    def handle(self):
        """
        Evaluates the answer against predefined evaluation metrics using an LLM.
        Based on the evaluation outcome and the type of question (main vs. sub-question), it:
        - Logs the reasoning trace into the agent's diary.
        - Marks the answer as final and successful if evaluation passes for the main user question.
        - Handles failed evaluations by logging errors, incrementing bad attempt counters, and analyzing what went wrong.
        - Records failed reasoning paths and resets internal step state for retry attempts.
        """
        LOGGER.info("Handling %s", self)

        self.references = self.clean_references(self.references)

        # first step and llm is confident to answer (without refs)
        if self.state.step == 1 and len(self.references) == 0:
            self.state.is_final = True
            self.state.stop_reason = AgentStopReason.TRIVIAL_ANSWER
            return

        eval_current_answer = {"pass": True, "think": ""}
        if (
            self.eval_answer
            and len(self.state.question_evals[self.state.current_question]) > 0
        ):
            eval_current_answer = self.evaluate_answer(
                question=self.state.current_question,
                answer=self.answer,
                knowledge_items=self.state.knowledge_items,
            )

        # Answered user question
        if self.state.current_question == self.state.user_query:
            if eval_current_answer["pass"]:
                self.state.steps_trace.append(
                    _MAIN_QUESTION_PASS_EVAL_DIARY.format(
                        step=self.state.step,
                        current_question=self.state.current_question,
                        answer=self.answer,
                        eval_current_answer_think=eval_current_answer["think"],
                    )
                )
                self.state.is_final = True
                self.state.stop_reason = AgentStopReason.FINAL_ANSWER_OK
                return
            else:
                if eval_current_answer["type"] == "strict":
                    self.state.final_answer_pip.append(
                        eval_current_answer.get("improvement_plan", "")
                    )
                    # remove strict
                    self.state.question_evals[self.state.current_question] = [
                        metric
                        for metric in self.state.question_evals[
                            self.state.current_question
                        ]
                        if metric != "strict"
                    ]

                if self.state.bad_attempts >= self.max_bad_attempts:
                    self.state.is_final = False
                    self.state.stop_reason = AgentStopReason.MAX_BAD_ATTEMPTS
                    return

                self.state.steps_trace.append(
                    _MAIN_QUESTION_FAIL_EVAL_DIARY.format(
                        step=self.state.step,
                        current_question=self.state.current_question,
                        answer=self.answer,
                        eval_current_answer_think=eval_current_answer["think"],
                    )
                )

                self.state.bad_attempts += 1
                self.bad_attempt = True

                error_analysis = self.analyze_steps()
                # Save trace of bad actions and reset the steps trace
                self.state.bad_actions.append(
                    {
                        "question": self.state.current_question,
                        "answer": self.answer,
                        "evaluation": eval_current_answer["think"],
                        "recap": error_analysis["recap"],
                        "blame": error_analysis["blame"],
                        "improvement": error_analysis["improvement"],
                    }
                )

                self.state.allow_answer = False  # Disable immediate answer
                self.state.steps_trace = []  # Reset the context
                self.state.step = 1  # reset to step 1 as the previous actions are forgotten from steps_trace

        elif eval_current_answer["pass"]:
            self.state.steps_trace.append(
                _SUB_QUESTION_PASS_EVAL_DIARY.format(
                    step=self.state.step,
                    current_question=self.state.current_question,
                    answer=self.answer,
                    eval_current_answer_think=eval_current_answer["think"],
                )
            )
            self.state.knowledge_items.append(
                KnowledgeItem(
                    type=KnowledgeItemType.FROM_ANSWER_STEP,
                    question=self.state.current_question,
                    answer=self.answer,
                    references=list(set(self.references)),
                    updated_at=get_current_datetime(),
                )
            )
