# build image (DHI dev variant, Python 3.11, Debian 13/Trixie)
# =============================================================
FROM dhi.io/python:3.11-debian13-dev AS build-image

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /opt/requirements.txt
RUN sed -i -re 's/^tensorflow\b/tensorflow-cpu/g' /opt/requirements.txt
RUN set -ex \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --target /opt/python -r /opt/requirements.txt \
    && python -m pip install --no-cache-dir --target /opt/python "fastapi" "uvicorn[standard]"


# hardened runtime image (DHI, Python 3.11, Debian 13/Trixie)
# ============================================================
FROM dhi.io/python:3.11-debian13 AS api

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/python

EXPOSE 4343
WORKDIR /opt/ncid

COPY --from=build-image /opt/python /opt/python
COPY --chown=nonroot:nonroot . /opt/ncid

CMD ["python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "4343"]
