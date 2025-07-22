from common.types import KnowledgeItem


def get_knowledge_item_default_xml_string(item: KnowledgeItem, idx: int = None) -> str:
    return f"""<knowledge{"-" + str(idx) if idx else ""}>
<question>
{item.question}
</question>
<answer>
{item.answer}
</answer>{"\n<url>" + item.references + "</url>" if (item.references and item.type == "from_visit_step") else ""}
</knowledge{"-" + str(idx) if idx else ""}>"""
