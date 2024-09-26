# # Stage 1: Base dependencies
# FROM python:3.9-slim AS base

# # Set the working directory
# WORKDIR /app

# COPY . /app

# # Upgrade pip and install dependencies
# RUN pip install --upgrade pip && \
#     pip install -r /app/requirements.txt

# FROM base AS final

# # Set PYTHONPATH environment variable
# ENV PYTHONPATH="/app:${PYTHONPATH}"


# Stage 1: Base dependencies
FROM python:3.9-slim AS base

# Set the working directory
WORKDIR /app

# Copy the rest of the code
COPY . /app


# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt


FROM base AS final

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/app:${PYTHONPATH}"
