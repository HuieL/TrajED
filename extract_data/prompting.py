import pathlib
from copy import deepcopy
from typing import List, Optional, Type, TypeVar
from dataclasses import dataclass

PROMPTS_ROOT = (pathlib.Path(__file__).parent).resolve()

T = TypeVar("T")

@dataclass(frozen=True)
class Item:
    trajectory: str
    comparison_trajectories: List[str]
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
        prompt_filename = "question.prompt"
    else:
        prompt_filename = "message.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    comprisons = task_item.comparison_trajectories

    # Format the potential categories into strings
    formatted_comprison_texts = []
    for neighbor_index, neighbor in enumerate(comprisons):
        formatted_neighbors = f"Person [{neighbor_index+1}]({neighbor}) "
        formatted_comprison_texts.append(
            formatted_neighbors
        )

    return_node_text = prompt_template.format(
            trajectory=task_item.trajectory,
            comparison_trajectories="\n".join(formatted_comprison_texts),
            )
    
    return return_node_text
