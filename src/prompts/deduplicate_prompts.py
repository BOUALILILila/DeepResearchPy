from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from llms.message import Message

DEDUP_QUERIES_SYS_PROMPT = """You are an expert in identifying when search queries mean the same thing. Given a list of queries, your job is to extract a subset that contains only unique queries—meaning they are not semantically redundant. This means removing queries that express the same intent or ask for the same information, even if they're worded differently.

<similarity-definition>
1. Consider semantic meaning and query intent, not just lexical similarity
2. Account for different phrasings of the same information need
3. Queries with same base keywords but different operators are NOT duplicates
4. Different aspects or perspectives of the same topic are not duplicates
5. Consider query specificity - a more specific query is not a duplicate of a general one
6. Search operators that make queries behave differently:
   - Different site: filters (e.g., site:youtube.com vs site:github.com)
   - Different file types (e.g., filetype:pdf vs filetype:doc)
   - Different language/location filters (e.g., lang:en vs lang:es)
   - Different exact match phrases (e.g., "exact phrase" vs no quotes)
   - Different inclusion/exclusion (+/- operators)
   - Different title/body filters (intitle: vs inbody:)
</similarity-definition>"""

env = Environment(
    loader=FileSystemLoader((Path(__file__).parent / "templates").as_posix())
)


def get_query_dedup_prompts(queries: list[str]) -> list[Message]:
    user_template = env.get_template("query_dedup_user_prompt_template.j2")
    user_content = user_template.render(queries=queries)

    return [
        Message(role="system", content=DEDUP_QUERIES_SYS_PROMPT),
        Message(role="user", content=user_content),
    ]
