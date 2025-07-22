from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from common.types import KnowledgeItem, SearchResult
from utils.date_utils import get_current_datetime

from .prompt_utils import get_knowledge_item_default_xml_string

env = Environment(
    loader=FileSystemLoader((Path(__file__).parent / "templates").as_posix())
)


def get_url_descriptor(result: SearchResult) -> str:
    output = f"[weight = {result.weight:.2f}] {result.url}: {result.title}"
    if len(result.description):
        output += f" - {result.description}"

    return output


def get_main_agent_prompt(
    knowledge_items: list[KnowledgeItem],
    action_history: list,
    bad_actions: list,
    available_actions: list,
    urls_to_visit: list,
    max_search_queries: int,
    max_decomposition_questions: int,
    enforce_answer: bool = False,
):
    template = env.get_template("main_agent_prompt_template.j2")

    action_sections = []

    if not enforce_answer:
        if "visit" in available_actions and urls_to_visit:
            urls = [get_url_descriptor(url) for url in urls_to_visit]
            visit_section = f"""
<action-visit>
- Crawl and read full content from URLs. You can get the fulltext of any URL
- You must check URLs mentioned in <question> if any
- Choose and visit relevant URLs below for more knowledge. Higher weight suggests more relevance:
<available-urls-to-visit>
- {"\n -".join(urls)}
</available-urls-to-visit>
</action-visit>"""
            action_sections.append(visit_section)

        if "search" in available_actions:
            action_sections.append(
                f"""
<action-search>
- Use web search to find relevant information
- Build diverse web search queries based on the intention of the original question and the expected answer format
- Always prefer a single search request, only add another request if the original question covers multiple aspects or elements and one query is not enough
- Each request should focus on one specific aspect of the original question
- Do not generate more than {max_search_queries} queries
- Do not generate multiple similar queries, queries should be diverse. If the topic is broad, generate more than 1 query, else 1 query is enough
</action-search>"""
            )

        if "answer" in available_actions:
            action_sections.append(
                """
<action-answer>
- Provide a high-quality **detailed** and accurate answer to the user's question based on the provided context
- For greetings, casual conversation, general knowledge questions, answer them directly
- For all other questions you MUST provide a verified answer with references to the gathered knowledge
- Each reference identifies the index of the knowledge item used
- Reference ONLY the information specified in <knowledge>. If a URL seems relevant, use <action-visit> to get its content first before using it to answer
- Be exhaustive in your answer
- If uncertain, use <action-reflect>
</action-answer>"""
            )

        if "reflect" in available_actions:
            action_sections.append(
                f"""
<action-reflect>
- Think slowly through the given context including <question>, <context>, <knowledge>, <bad-attempts>, and <learned-strategy> to identify knowldege gaps or areas that need deeper exploration
- If you identify gaps, generate a list of relevant clarifying questions that deeply related to the original question and lead to the answer
- Do not generate more than {max_decomposition_questions}
</action-reflect>"""
            )

    rendered_prompt = template.render(
        current_date=get_current_datetime(),
        knowledge_items=[
            get_knowledge_item_default_xml_string(k, i)
            for i, k in enumerate(knowledge_items, start=1)
        ],
        action_history=action_history,
        bad_actions=bad_actions,
        enforce_answer=enforce_answer,
        action_sections=action_sections,
    )

    return rendered_prompt
