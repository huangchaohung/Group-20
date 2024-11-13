# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip, setuptools, and wheel - this can help avoid some build issues
RUN pip install --upgrade pip setuptools wheel

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory and data directory into the container
COPY src/ /app/src/
COPY data/ /app/data/

# Set environment variables for Flask and Python path
ENV FLASK_APP=/app/src/main.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app/src

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]
