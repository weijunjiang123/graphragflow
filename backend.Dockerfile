# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# (e.g., for cryptography or lxml) - keep this minimal
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libpq-dev \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ /app/src/

# Copy .env.example if it exists and is needed for default non-sensitive configs
# For this project, config.py handles defaults if env vars are not set.
# If .env.example is used, ensure it does not contain secrets.
# COPY .env.example .env

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# The API main is located at src/api/main.py, with the FastAPI app instance named 'app'
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
