from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str
