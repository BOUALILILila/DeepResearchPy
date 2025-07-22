from abc import ABC, abstractmethod

from common.types import ResearchState


class BaseStep(ABC):
    def __init__(self, state: ResearchState) -> None:
        self.state = state

    @abstractmethod
    def handle(self) -> None:
        raise NotImplementedError()

    def as_markdown(self):
        raise NotImplementedError()
