# FROM python:3.9-slim

# WORKDIR /app

# COPY . /app

# RUN pip install --upgrade pip

# RUN pip install -r /app/requirements.txt

# ENV PYTHONPATH="/app:${PYTHONPATH}"



# Stage 1: Base dependencies
FROM python:3.9-slim AS base

# Set the working directory
WORKDIR /app

COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Install necessary system packages for opencv-python
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Application code
FROM base AS final

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/app:${PYTHONPATH}"
