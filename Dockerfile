# Evaluator-side image for running UNTRUSTED agents with an OS-level boundary (see README, "Isolation").
# The agent is started by the runner as:  docker run -i --rm --network none --read-only ...  (traceforge.runner.container)
FROM python:3.12-slim
WORKDIR /opt/traceforge
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e . && useradd -u 10001 -M agent
USER 10001
ENTRYPOINT ["python"]
