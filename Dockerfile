# base python image version
ARG PYTHON_VERSION=3.11-slim-trixie


# base image
# ============
FROM python:${PYTHON_VERSION} AS base

# configure python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# setup virtualenv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# build image
# =============
FROM base AS build-image

# install dependencies
COPY requirements.txt /opt/requirements.txt
RUN sed -i -re 's/^tensorflow\b/tensorflow-cpu/g' /opt/requirements.txt
RUN set -ex \
    && python -m venv --symlinks --clear "$VIRTUAL_ENV" \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/requirements.txt \
    && pip install --no-cache-dir "fastapi" "uvicorn[standard]"


# api image
# ======================
FROM base AS api

EXPOSE 4343

# copy virtualenv
COPY --from=build-image "$VIRTUAL_ENV" "$VIRTUAL_ENV"

# create source directory
RUN mkdir -p /opt/ncid/
WORKDIR /opt/ncid

# change to non-root user
RUN useradd apiuser
USER apiuser

COPY --chown=apiuser:apiuser . /opt/ncid

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "4343"]
