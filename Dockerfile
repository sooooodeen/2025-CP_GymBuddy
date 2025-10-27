# Dockerfile

# Use a standard Python image based on Debian
FROM python:3.10-slim

# Install system dependencies for MediaPipe and OpenCV
# This is crucial for running computer vision/AI libraries on a headless server.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    # Clean up afterwards to keep the image small
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/src/app

# Copy requirements and install dependencies first for better Docker layer caching
COPY requirements.txt ./
# The --no-cache-dir flag saves disk space after installation
# NOTE: This step will take a LONG time and consume a lot of RAM during the BUILD phase
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (your app.py, models, static files, etc.)
COPY . .

# Create a non-root user for security best practice
RUN useradd -m appuser
USER appuser

# Set the final run command
# Gunicorn is used for production-grade serving with eventlet workers for Flask-SocketIO
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:8000", "app:app"]
# REMINDER: Ensure 'app:app' is the correct module:application name for your Flask app