# Tessellate AI inference tier — Hugging Face Space / any container host.
# Build context is the repository root:  docker build -t tessellate-space .

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/hf_cache

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY core/ core/
COPY api/ api/
COPY ui/ ui/
COPY data/track_constants.json data/
COPY data/fixtures/ data/fixtures/

# Circuit memory plus the replay fixture are in the repo; only the compiled
# weights come from the Hub model repo.
COPY fetch_space_weights.py ./
RUN python fetch_space_weights.py

RUN chmod -R a+w /app/data /app/models

COPY docker-entrypoint.sh ./
EXPOSE 8000
ENTRYPOINT ["/bin/sh", "./docker-entrypoint.sh"]
