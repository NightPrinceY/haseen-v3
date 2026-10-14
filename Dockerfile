# HaSeen 3 on CPU: the haseen package + HaSeen-3-ONNX (int8), no PyTorch. The model is baked in at build time
# (a token is only needed while the model repo is private: pass it as a build secret), so the container runs offline.
#   DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t haseen3 .
#   docker run --rm -p 8000:8000 haseen3                                   # POST /detect, GET /health
#   docker run --rm haseen3 "أبغى أساعدك، سوف أحوّلك للقسم المختص الحين" -t egy --json
FROM python:3.11-slim
LABEL org.opencontainers.image.title="HaSeen 3" \
      org.opencontainers.image.description="Word-level Arabic dialect deviation detection, 6 targets (MSA, Egyptian, Saudi, Gulf, Levantine, Iraqi), CPU, offline" \
      org.opencontainers.image.version="3.0.0" \
      org.opencontainers.image.source="https://github.com/NightPrinceY/haseen-v3"
ENV HF_HOME=/opt/haseen/hf PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=secret,id=hf_token \
    pip install . && HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null)" haseen download \
    && useradd --system --create-home --uid 10001 haseen && chmod -R a+rX /opt/haseen
ENV HF_HUB_OFFLINE=1
USER haseen
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
ENTRYPOINT ["haseen"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
