from datetime import datetime
from pathlib import Path
from inspect import Parameter
import inspect
import os
from openai import OpenAI

ROOT_PATH = Path(__file__).parent.parent.absolute()

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
MODE = os.environ.get("MODE", "")

client = OpenAI(api_key=AIPROXY_TOKEN, base_url="https://aiproxy.sanand.workers.dev/openai/v1/")

def get_path(input_path: str) -> Path:
    if MODE == "debug": 
        return Path(f"{str(ROOT_PATH) + input_path}")
    print(f"{input_path=}")
    return Path(input_path)


# Reference function_to_json from OpenAI's Swarm Library, MIT License
def function_to_chat_completion_params_tools(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "strict": True,  # for structured decoding I think
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
                "additionalProperties": False
            },
        },
    }

