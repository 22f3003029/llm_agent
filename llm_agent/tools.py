import subprocess
from .utils import function_to_chat_completion_params_tools, get_path, DEBUG
from datetime import datetime

def parse_dates(datestr: str):
    return datetime(datestr)


def run_prettier(file_path: str, prettier_version: str):
    """Run a specific version of prettier on the given file path

    The file should be of form
    """
    print(f"{file_path=}")
    print(f"{prettier_version=}")
    file_path = get_path(input_path=file_path, debug=DEBUG)
    subprocess.run(["npx", "-y", f"prettier@{prettier_version}", file_path, "--write"])


tools = {
    "run_prettier": {

        "tool": function_to_chat_completion_params_tools(run_prettier),
        "func": run_prettier,
    }
}
