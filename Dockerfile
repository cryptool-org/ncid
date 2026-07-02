# build image (DHI dev variant, Python 3.11, Debian 13/Trixie)
# =============================================================
FROM dhi.io/python:3.11-debian13-dev AS build-image

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /opt/requirements.txt
RUN sed -i -re 's/^tensorflow=/tensorflow-cpu=/g' /opt/requirements.txt

RUN set -ex \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /opt/requirements.txt \
    && python -m pip install --no-cache-dir "fastapi" "uvicorn[standard]"

# hardened runtime image (DHI, Python 3.11, Debian 13/Trixie)
# ============================================================
FROM dhi.io/python:3.11-debian13 AS api

LABEL org.opencontainers.image.description="Runtime of the Neural Cipher IDentifier HTTP API"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 4343
WORKDIR /opt/ncid

COPY --from=build-image /opt/venv /opt/venv
COPY --chown=nonroot:nonroot . /opt/ncid

CMD ["python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "4343"]
