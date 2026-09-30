# Container for the Hugging Face Space (also works with any Docker host)
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY app/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# app/model/ must exist (run python app/train.py first); deploy/deploy_space.py uploads it
COPY --chown=user app/ ./app/

USER user
ENV HOST=0.0.0.0 PORT=7860 BLOCK_PRIVATE_URLS=1 HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["python", "app/server.py"]
