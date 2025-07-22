from common.cherry_picker import CherryPicker
from common.exceptions import CouldNotReadUrl
from common.types import KnowledgeItem, KnowledgeItemType
from utils.logger import get_logger
from utils.url_utils import get_url_content_as_markdown

from .base_step import BaseStep

LOGGER = get_logger(__name__, step="VISIT")

_VISIT_URL_SUCCESS_DIARY = """
At step {step}, you took the **visit** action and deep dive into the following URLs:
{formatted_visited_urls}
You found some useful information on the web and add them to your knowledge for future reference.
"""

_COULD_NOT_VISIT_URLS_DIARY = """
At step {step}, you took the **visit** action and try to visit some URLs but failed to read the content. You need to think out of the box or cut from a completely different angle.
"""

_NO_NEW_URLS_TO_VISIT = """s
At step {step}, you took the **visit** action. But then you realized you have already visited these URLs and you already know very well about their contents.
You decided to think out of the box or cut from a completely different angle.
"""


class VisitStep(BaseStep):
    """
    Handles a visit action.
    Reads the content of a URL and retrieves the most relavant text snippets from the raw text based on semantic similarity. Records the trace in the agent's diary.
    """

    def __init__(
        self, state, urls, cherry_picker: CherryPicker, max_urls_per_step: int = 4
    ):
        super().__init__(state=state)
        self.urls = urls
        self.max_urls_per_step = max_urls_per_step
        self.cherry_picker = cherry_picker

    def __repr__(self):
        return f"VisitStep(step={self.state.step}, current_question={self.state.current_question}, urls={self.urls}, max_urls_per_step={self.max_urls_per_step})"

    def as_markdown(self):
        if len(self.urls) == 0:
            return None
        f_urls = "\n- ".join(self.urls)
        return f"""Visiting urls:\n- {f_urls}"""

    def visit_urls(self, urls: list[str]) -> tuple[list[str], list[str]]:
        visited_urls, bad_urls = [], []
        for url in urls:
            LOGGER.info("Visiting URL: %s", url)
            try:
                content = get_url_content_as_markdown(url=url)
                LOGGER.debug(
                    "Cherry picking snippets from content with length %d (chars)",
                    len(content),
                )
                cherry_picked_content = self.cherry_picker.cherry_pick(
                    question=self.state.current_question,
                    text=content,
                )
                self.state.knowledge_items.append(
                    KnowledgeItem(
                        type=KnowledgeItemType.FROM_VISIT_STEP,
                        question=f'What do experts say about "{self.state.current_question}"?',
                        answer=cherry_picked_content,
                        references=url,
                    )
                )
                visited_urls.append(url)
            except CouldNotReadUrl:
                bad_urls.append(url)

        LOGGER.info("Visited urls: %s", visited_urls)
        LOGGER.info("Bad urls: %s", bad_urls)
        return visited_urls, bad_urls

    def filter_urls(self, urls: list[str]) -> list[str]:
        return [
            url
            for url in urls
            if url.startswith("http") and url not in self.state.visited_urls
        ][: self.max_urls_per_step]

    def handle(self):
        LOGGER.info("Handling %s", self)

        self.urls = self.filter_urls(self.urls)

        if len(self.urls) > 0:
            visited_urls, bad_urls = self.visit_urls(urls=self.urls)
            self.state.visited_urls.extend(visited_urls)  # Keep track of visited urls
            self.state.all_urls = [
                url
                for url in self.state.all_urls
                if url not in bad_urls  # filter out bad urls
            ]

            if len(visited_urls) > 0:
                self.state.steps_trace.append(
                    _VISIT_URL_SUCCESS_DIARY.format(
                        step=self.state.step,
                        formatted_visited_urls=f"- {'\n- '.join(self.state.visited_urls)}",
                    )
                )
            else:  # no visited_url
                self.state.steps_trace.append(
                    _COULD_NOT_VISIT_URLS_DIARY.format(step=self.state.step)
                )
        else:  # no URL at all
            self.state.steps_trace.append(
                _NO_NEW_URLS_TO_VISIT.format(step=self.state.step)
            )
        self.state.allow_visit = False
