from langchain_core.prompt_values import StringPromptValue
from typing import Any

def to_str_safe(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, StringPromptValue):
        return obj.to_string()
    if hasattr(obj, "to_string") and callable(obj.to_string):
        return obj.to_string()
    if hasattr(obj, "to_str") and callable(obj.to_str):
        return obj.to_str()
    return str(obj)
