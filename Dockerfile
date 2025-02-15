FROM python:3.12-slim-bookworm

WORKDIR /llm_agent
# Copy the project into the image
ADD . /llm_agent

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN apt-get install -y nodejs
RUN apt-get install -y npm
RUN npm install

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Sync the project into a new environment, using the frozen lockfile
RUN uv sync --frozen

ENV PATH="/llm_agent/.venv/bin/:$PATH"
# Setup an app user so the container doesn't run as the root user
# RUN useradd llm_agent
# USER llm_agent

CMD ["python", "-m", "uvicorn", "llm_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]