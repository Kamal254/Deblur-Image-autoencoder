FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="/app:${PYTHONPATH}"