import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from common.config import Configuration
from common.types import AgentStopReason, KnowledgeItem, KnowledgeItemType
from deep_research.answer_step import AnswerStep
from deep_research.main_agent import DeepResearch

logging.disable(logging.CRITICAL)

step_titles = {
    "SearchStep": "Searching the Web",
    "VisitStep": "Reading URL(s)",
    "AnswerStep": "Answering",
    "ReflectStep": "Thinking",
}


def terminal_gui(agent: DeepResearch):
    """
    Launches a terminal-based GUI for interacting with a DeepResearch agent.

    Prompts the user for a query, processes it using the agent, and displays
    the steps taken by the agent and the final answer in a formatted terminal interface.
    """
    console = Console()
    console.clear()
    console.rule("[bold cyan]Welcome to Deep Research GUI")
    query = Prompt.ask("Ask your question")

    with console.status("", spinner="dots"):
        for current_step, is_final in agent(user_query=query):
            if is_final:
                reason = stop_reason(agent.state.stop_reason)
                console.rule(f"[bold green]Final Answer{reason}")
                answer = format_final_answer_as_markdown(
                    agent=agent, final_answer_step=current_step
                )
                md = Markdown(answer)
                console.print(md)

                console.print(
                    Panel("[bold green]:tada: All steps completed!", style="green")
                )

            else:
                title = step_titles[current_step.__class__.__name__]
                step_desc = current_step.as_markdown()
                if step_desc:
                    md = Markdown(step_desc)
                    console.rule(f"[bold blue]{title}")
                    console.print(md)


def format_final_answer_as_markdown(
    agent: DeepResearch, final_answer_step: AnswerStep
) -> str:
    if len(agent.state.knowledge_items) > 0:
        sources = format_knowledge(knowledge_items=agent.state.knowledge_items)
        return f"{final_answer_step.answer}\n\n\n# Sources\n\n{sources}"
    return f"{final_answer_step.answer}"


def format_knowledge_item(item: KnowledgeItem) -> str:
    if item.type == KnowledgeItemType.FROM_VISIT_STEP:
        return f"""Visited URL {item.references}"""
    if item.type == KnowledgeItemType.FROM_ANSWER_STEP:
        return f"""Sub-question {item.question}

{item.answer}

## Anwer References: {item.references}"""


def format_knowledge(knowledge_items: list[KnowledgeItem]) -> str:
    output = ""
    for idx, item in enumerate(knowledge_items, start=1):
        output += f"\n\n[{idx}] {format_knowledge_item(item)}"
    return output


def stop_reason(reason: AgentStopReason) -> str:
    if reason == AgentStopReason.MAX_BAD_ATTEMPTS:
        return " : Maximum Bad attempts exceeded"
    if reason == AgentStopReason.MAX_TOKENS_BUDGET:
        return " : Token budget exhausted"
    return ""


# Run it
def run_gui(config_path: os.PathLike | Path):
    try:
        config = Configuration.from_yaml(path=config_path)
        deep_research = DeepResearch(config=config)
        terminal_gui(deep_research)
    except KeyboardInterrupt:
        Console().print("\n[bold red]Exited by user.[/bold red] 👋")
        sys.exit(0)
