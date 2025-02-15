from fastapi import FastAPI
import os
from openai import OpenAI
import subprocess
from .tools import tools
from .utils import get_path, DEBUG

"""
During Setup:

- Install npm to make npx work

"""

app = FastAPI()

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]

client = OpenAI(api_key=AIPROXY_TOKEN, base_url="https://aiproxy.sanand.workers.dev/openai/v1/")

def get_params(tools: list[dict], messages: list[dict]) -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": messages,
        "tools": tools,
        "stream": False,
    }

def get_completion(task: str):

    params = get_params(tools=[tool["tool"] for tool in tools.values()], messages=[{"role": "user", "content": task}])
    return client.chat.completions.create(
        **params
    )

import json

def handle_completion(completion):
    tool_calls = completion.choices[0].message.tool_calls
    print(f"{tool_calls=}")
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        func = tools[func_name]["func"]
        func(**func_args)
    # no return

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/run")
async def run(task: str):
    completion = get_completion(task=task)
    print(f"{completion=}")
    handle_completion(completion)
    return 200


@app.get("/read")
async def read(path: str):
    path = get_path(input_path=path, debug=DEBUG)

    if path.exists():
        with open(path, "r") as f:
            return f.read()
    else:
        return 404