import subprocess
from .utils import function_to_chat_completion_params_tools, get_path, MODE, client, AIPROXY_TOKEN
from datetime import datetime
from dateutil.parser import parse
import json
import os
from pathlib import Path
from typing import Literal 
import numpy as np
import git
from bs4 import BeautifulSoup

import pandas as pd

from PIL import Image
import requests
import base64
import sqlite3


def parse_dates(datestr: str):
    return datetime(datestr)


def run_prettier(file_path: str, prettier_version: str):
    """Run a specific version of prettier on the given file path

    The file should be of form
    """
    print(f"{file_path=}")
    print(f"{prettier_version=}")
    file_path = get_path(input_path=file_path)
    subprocess.run(["npx", "-y", f"prettier@{prettier_version}", file_path, "--write"])

def run_script_from_source(source_file: str, email: str):
    """
    Run the script from source for data generation.
    """
    print(f"{source_file}")
    print(f"{email}")
    subprocess.run(["uv", "run", source_file, email])



def count_weekdays(input_path: str, output_path: str, weekday: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]):
    """Count the number of specific weekdays in the list of dates"""
    input_path = get_path(input_path=input_path)
    with open(input_path, "r") as f:
        dates = f.readlines() 

    weekdays = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    count = sum(1 for date in dates if parse(date.strip()).weekday() == weekdays[weekday])

    output_path = get_path(input_path=output_path)

    with open(output_path, "w") as f:
        f.write(str(count))


def sort_by_keys(input_path: str, output_path: str, keys: list[str]):
    """Sort the json data in input file data by keys. 

    Provide the keys in a list of strings, such as ["first_name", "last_name"]
    """
    input_path = get_path(input_path=input_path)
    output_path = get_path(input_path=output_path)
    with open(input_path, "r") as f:
        data = json.load(f)
    keys = json.loads(keys)
    data = sorted(data, key=lambda x: tuple(x[key] for key in keys))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def process_recent_logfiles(dir_path: str, output_path: str, num_logs: int):
    """
    Process recent log files and write the first lines in the given output path.

    Provide the directory path of the log files in dir_path and the output file name
    in output_path
    """
    dir_path = get_path(input_path=dir_path)
    output_path = get_path(input_path=output_path)

    files = list(dir_path.glob("*.log"))
    files = list(sorted(files, key=os.path.getmtime, reverse=True))
    files = files[:num_logs]

    first_lines = []
    for file in files:
        with open(file, "r") as f:
            first_lines.append(f.readline().strip())

    with open(output_path, "w") as f:
        f.write("\n".join(first_lines))


def get_markdown_heading(input_path: Path):
    with open(input_path, "r") as f:
        for line in f.readlines():
            if line.startswith("# "):

                return line[2:].strip()


def create_index_from_markdown_files(dir_path: str, output_path: str):
    """
    Create index, a mapping from the name of the file to its location from markdown files
    directory.
    """
    dir_path = get_path(input_path=dir_path)
    output_path = get_path(input_path=output_path)

    files = list(dir_path.glob("**/*.md"))
    index = {}
    for file in files:
        heading = get_markdown_heading(file)
        index[str(file.relative_to(get_path("/data/docs/")))] = heading
    print(index)
    with open(output_path, "w") as f:
        json.dump(index, f)



def extract_senders_email(input_path: str, output_path: str):
    """
    Extract the email from the given input path and save it to output path
    """
    input_path = get_path(input_path=input_path)
    output_path = get_path(input_path=output_path)

    with open(input_path, "r") as f:
        text = f.read()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the sender's email address from the following email. Respond with only the email address."},
            {"role": "user", "content": text}
        ]
    )

    email = response.choices[0].message.content.strip()
    with open(output_path, "w") as f:
        f.write(email)




def extract_credit_card_number(input_path: str, output_path: str):
    """Extract credit card number from image using LLM"""
    
    input_path = get_path(input_path)
    output_path = get_path(output_path)
    
    with open(input_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "There are multiple numbers in the image. Extract the main longest sequence of number. Return only the number without spaces."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    )
    
    card_number = response.choices[0].message.content.strip().replace(" ", "")
    with open(output_path, 'w') as f:
        f.write(card_number)


def get_similar_comments(input_path: str, output_path: str):
    """Take comments from the input file and save the most similar comments on the output path"""
    input_path = get_path(input_path)
    output_path = get_path(output_path)
    
    with open(input_path, "r") as f:
        comments = f.read().split("\n")

    data = client.embeddings.create(input=comments, model="text-embedding-3-small")
    print(data)
    embeddings = np.array([np.array(list(d.embedding)) for d in data.data])

    similarity = np.dot(embeddings, embeddings.T)
    # Create mask to ignore diagonal (self-similarity)
    np.fill_diagonal(similarity, -np.inf)
    # Get indices of maximum similarity
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    print(i, j)
    output = "\n".join(sorted([comments[i], comments[j]]))
    with open(output_path, "w") as f:
        f.write(output)


def get_total_sales(input_path: str, ticket_type: str, output_path: str):
    """
    Calculate total sales for specific ticket type
    
    input_path is the db file path

    """
    input_path = get_path(input_path)
    output_path = get_path(output_path)
    
    conn = sqlite3.connect(input_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT SUM(units * price)
        FROM tickets
        WHERE type = ?
    """, (ticket_type,))
    
    total = cursor.fetchone()[0] or 0
    conn.close()
    
    with open(output_path, 'w') as f:
        f.write(str(total))    




def fetch_api_data(api_url: str, output_path: str, headers: dict = None):
    """Fetch some data given the api url and headers and save it on output path"""
    output_path = get_path(output_path)
    
    response = requests.get(api_url, headers=headers, timeout=30)
    response.raise_for_status()
    
    with open(output_path, 'w') as f:
        json.dump(response.json(), f, indent=2)

def clone_git_and_commit(repo_url: str, output_dir_path: str, commit_message: str):
    """Clone reposiroty and make a commit"""
    
    output_dir_path = get_path(output_dir_path)

    # Clone repository
    repo = git.Repo.clone_from(repo_url, output_dir_path)
    
    # Make changes (example)
    repo.index.add('*')
    repo.index.commit(commit_message)
        


def run_query_in_database(query: str, db_file: str, output_path: str):
    """Run SQL query and save results"""
    
    db_file = get_path(db_file)
    output_path = get_path(output_path)

    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df.to_json(output_path, orient='records', indent=2)


def scrape_website(url: str, output_path: str):
    """Scrape website data"""
    
    output_path = get_path(output_path)
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    data = {
        'title': soup.title.string if soup.title else None,
        'text': soup.get_text(),
        'links': [{'text': a.text, 'href': a.get('href')} for a in soup.find_all('a', href=True)]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_image(image_path: str, output_path: str, max_size: tuple):
    """Process image (resize/compress)"""
    
    input_path = get_path(image_path)
    output_path = get_path(output_path)
    
    with Image.open(input_path) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize maintaining aspect ratio
        img.thumbnail(max_size)
        
        # Save with optimization
        img.save(output_path, 'JPEG', quality=85, optimize=True)


tools = {
    "run_prettier": {
        "tool": function_to_chat_completion_params_tools(run_prettier),
        "func": run_prettier,
    },
    "run_script_from_source": {
        "tool": function_to_chat_completion_params_tools(run_script_from_source),
        "func": run_script_from_source,
    },
    "count_weekdays": {
        "tool": function_to_chat_completion_params_tools(count_weekdays),
        "func": count_weekdays,
    },
    "sort_by_keys": {
        "tool": function_to_chat_completion_params_tools(sort_by_keys),
        "func": sort_by_keys,
    },
    "create_index_from_markdown_files": {
        "tool": function_to_chat_completion_params_tools(create_index_from_markdown_files),
        "func": create_index_from_markdown_files,
    },
    "process_recent_logfiles": {
        "tool": function_to_chat_completion_params_tools(process_recent_logfiles),
        "func": process_recent_logfiles,
    },
    "extract_senders_email": {
        "tool": function_to_chat_completion_params_tools(extract_senders_email),
        "func": extract_senders_email,
    },
    "extract_credit_card_number": {
        "tool": function_to_chat_completion_params_tools(extract_credit_card_number),
        "func": extract_credit_card_number,
    },
    "get_total_sales": {
        "tool": function_to_chat_completion_params_tools(get_total_sales),
        "func": get_total_sales,
    },
    "get_similar_comments": {
        "tool": function_to_chat_completion_params_tools(get_similar_comments),
        "func": get_similar_comments,
    },
    # "fetch_api_data": {
    #     "tool": function_to_chat_completion_params_tools(fetch_api_data),
    #     "func": fetch_api_data,
    # },
    "clone_git_and_commit": {
        "tool": function_to_chat_completion_params_tools(clone_git_and_commit),
        "func": clone_git_and_commit,
    },
    "run_query_in_database": {
        "tool": function_to_chat_completion_params_tools(run_query_in_database),
        "func": run_query_in_database,
    },
    "scrape_website": {
        "tool": function_to_chat_completion_params_tools(scrape_website),
        "func": scrape_website,
    },
    "process_image": {
        "tool": function_to_chat_completion_params_tools(process_image),
        "func": process_image,
    },
}
