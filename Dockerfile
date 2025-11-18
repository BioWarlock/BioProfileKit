# 1) Base Image – Python 3.13 (wie im Projekt vorgesehen)
FROM python:3.13-slim AS runtime

# 2) Python / Pip Settings (optional, aber sinnvoll)
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on

# 3) Systempakete für C-Extensions (Cython, numpy, scipy, deine .pyx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 4) Arbeitsverzeichnis im Container
WORKDIR /app

# 5) Zuerst nur requirements kopieren und installieren (besseres Layer-Caching)
COPY requirements.txt ./

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# 6) Jetzt den kompletten BioProfileKit-Code in den Container kopieren
COPY . .

# 7) BioProfileKit installieren
# Statt pip + setup.py direkt:
RUN make install

# 8) Standard-EntryPoint: CLI 'bioprofilekit'
ENTRYPOINT ["bioprofilekit"]

# Optional: Standard-Command, z.B. Hilfe anzeigen
CMD ["--help"]