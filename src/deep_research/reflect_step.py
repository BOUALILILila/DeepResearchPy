from common.deduplicate_queries import DeduplicateQueries
from utils.logger import get_logger
from utils.sample_k import sample_k

from .base_step import BaseStep

LOGGER = get_logger(__name__, step="REFLECT")


class ReflectStep(BaseStep):
    """
    Handles a reflexion action.
    Deduplicates the sub-questions generated from the original user question and records the trace in the agent's diary.
    """

    def __init__(
        self,
        questions_to_answer: list[str],
        state,
        question_deduplicator,
        max_questions_to_answer: int = 5,
    ):
        super().__init__(state=state)
        self.questions_to_answer = questions_to_answer
        self.max_questions_to_answer = max_questions_to_answer

        self.question_deduplicator: DeduplicateQueries = question_deduplicator

    def __repr__(self):
        return f"ReflectStep(step={self.state.step}, questions_to_answer={self.questions_to_answer}, max_questions_to_answer={self.max_questions_to_answer})"

    def as_markdown(self):
        f_questions = "\n- ".join(self.questions_to_answer)
        return f"""Reflecting on questions: \n- {f_questions}"""

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

    def handle(self):
        LOGGER.info("Handling %s", self)

        self.questions_to_answer = self.deduplicate_questions(
            all_questions=self.state.all_questions,
            current_questions=self.questions_to_answer,
            k=self.max_questions_to_answer,
        )
        new_gap_questions = self.questions_to_answer

        if len(new_gap_questions) > 0:
            self.state.steps_trace.append(
                f"""
At step {self.state.step}, you took **reflect** and think about the knowledge gaps. You found some sub-questions are important to the question: "{self.state.current_question}"
You realize you need to know the answers to the following sub-questions:
- {"\n- ".join(new_gap_questions)}

You will now figure out the answers to these sub-questions and see if they can help you find the answer to the original question.
"""
            )
            self.state.gaps.extend(new_gap_questions)
            self.state.all_questions.extend(new_gap_questions)
        else:
            self.state.steps_trace.append(
                f"""
At step {self.state.step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "{self.state.current_question}" into gap-questions like this: ${", ".join(new_gap_questions)} 
But then you realized you have asked them before. You decided to to think out of the box or cut from a completely different angle.
"""
            )
            self.state.allow_reflect = False
