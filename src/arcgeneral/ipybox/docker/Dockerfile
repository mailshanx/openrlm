FROM python:3.14-slim-bookworm


ENV HOME=/root

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

WORKDIR ${HOME}

COPY pyproject.toml .python-version ${HOME}/
RUN uv sync --no-dev --no-install-project

COPY kernel.py /kernel.py

WORKDIR /app
ENV PATH="/root/.venv/bin:$PATH"
CMD ["python", "/kernel.py"]
