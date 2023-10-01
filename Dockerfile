# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:23.09-py3

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/cache/huggingface
