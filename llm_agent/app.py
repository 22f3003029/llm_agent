from fastapi import FastAPI, HTTPException
import os
from openai import OpenAI
import subprocess
from .tools import tools
from .utils import get_path, MODE, client, AIPROXY_TOKEN
from fastapi.responses import PlainTextResponse
"""
During Setup:

- Install npm to make npx work

"""

app = FastAPI()

def get_params(tools: list[dict], messages: list[dict]) -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": messages,
        "tools": tools,
        "stream": False,
    }

def get_completion(task: str):

    params = get_params(tools=[tool["tool"] for tool in tools.values()], 
        messages=[
        {"role": "system", "content": """
You are a smart intellegent assistant with tools to handle the user's request. 

"""},
        {"role": "user", "content": task}

        ])
    return client.chat.completions.create(
        **params
    )

import json

def handle_completion(completion):
    tool_calls = completion.choices[0].message.tool_calls
    print(f"{tool_calls=}")
    for tool_call in tool_calls[:1]:  # only do the first call, and not others
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


@app.get("/read", response_class=PlainTextResponse)
async def read(path: str):
    path = get_path(input_path=path)

    if path.exists():
        with open(path, "r") as f:
            return f.read()
    else:
        print(f"{MODE=}, {path=}")
        raise HTTPException(status_code=404)