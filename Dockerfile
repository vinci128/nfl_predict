FROM python:3.12-slim

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first — this layer is cached unless pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# Copy source code and install the project
COPY src/ ./src/
RUN uv sync --no-dev --frozen

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# data/, outputs/, and models/ are mounted as volumes at runtime — not baked in
CMD ["uvicorn", "nfl_predict.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
