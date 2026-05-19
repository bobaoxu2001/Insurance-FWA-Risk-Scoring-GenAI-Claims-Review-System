# Multi-stage Dockerfile for the Healthcare Provider FWA project.
#
# Build:
#   docker build -t fwa-portfolio .
#
# Run (Streamlit dashboard, port 8501):
#   docker run -p 8501:8501 fwa-portfolio
#
# Run a one-shot pipeline command:
#   docker run --rm fwa-portfolio python src/oig_leie_analysis.py
#
# Run with real data mounted from host (avoids re-downloading 0.5 GB each build):
#   docker run -v $(pwd)/data:/app/data -p 8501:8501 fwa-portfolio
#
# Notes
# -----
# - The base image is python:3.11-slim — smallest official Python image that
#   carries the libstdc++ scikit-learn needs.
# - We do NOT install torch / transformers in the default build; the tier-2
#   semantic-LLM path is opt-in via:
#       docker build --build-arg INSTALL_LLM=1 -t fwa-portfolio-llm .
# - Raw datasets (Kaggle, OIG LEIE, CMS) are NOT copied into the image. Mount
#   them at runtime or run scripts/download_real_data.sh inside the container.

FROM python:3.11-slim AS base

# System deps for matplotlib backend and pandas wheels on slim base
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: core deps (changes rarely → cacheable)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Layer 2 (optional): LLM deps. Build with --build-arg INSTALL_LLM=1 to enable.
ARG INSTALL_LLM=0
COPY requirements-llm.txt /app/
RUN if [ "$INSTALL_LLM" = "1" ]; then \
        pip install --no-cache-dir -r requirements-llm.txt; \
    fi

# Layer 3: project code
COPY src/      /app/src/
COPY scripts/  /app/scripts/
COPY tests/    /app/tests/
COPY app.py config.py Makefile /app/
COPY data/documents/ /app/data/documents/
# README.md last so changes to it do not bust earlier layers
COPY README.md /app/

# Streamlit defaults
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
