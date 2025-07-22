from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from llms.message import Message
from utils.date_utils import get_current_datetime

env = Environment(
    loader=FileSystemLoader((Path(__file__).parent / "templates").as_posix())
)


def get_query_rewrite_prompts(
    query: str, think: str, initial_search_results: list[str]
) -> list[Message]:
    system_template = env.get_template("query_rewrite_sys_prompt_template.j2")
    user_template = env.get_template("query_rewrite_user_prompt_template.j2")

    system_content = system_template.render(current_datetime=get_current_datetime())
    user_content = user_template.render(
        query=query, think=think, search_results=initial_search_results
    )

    return [
        Message(role="system", content=system_content),
        Message(role="user", content=user_content),
    ]
