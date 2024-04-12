import pathlib
from copy import deepcopy
from typing import List, Optional, Type, TypeVar
from dataclasses import dataclass

PROMPTS_ROOT = (pathlib.Path(__file__).parent).resolve()

T = TypeVar("T")

@dataclass(frozen=True)
class Item:
    trajectory: str
    index: List[int]
    id: Optional[str] = None
    content: Optional[str] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Item from dict.")
        id = data.pop("id", None)
        return cls(**dict(data, id=id))

def get_prompt(
    task_item: Item,
    task_name: str,
):
    if task_name == "question":
        prompt_filename = "nola_question.prompt"
    else:
        prompt_filename = "nola.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    return_node_text = prompt_template.format(
            trajectory=task_item.trajectory,
            index=task_item.index,
            )
    
    return return_node_text
