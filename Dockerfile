FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

COPY pyproject.toml README.md Makefile ./

RUN pip install --no-build-isolation -e . --no-deps || true && \
    pip install $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))")

COPY . .

RUN make install

ENTRYPOINT ["bioprofilekit"]

CMD ["--help"]