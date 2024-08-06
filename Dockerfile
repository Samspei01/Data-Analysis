# Use the official Python 3.11 image
FROM python:3.11.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to ensure we are using the latest version for Python 3.11
RUN python3.11 -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt ./
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Check pip version
RUN python3.11 -m pip --version

# Make port 5003 available to the world outside this container
EXPOSE 5003

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the command to start the application
CMD ["python3.11", "-m", "flask", "run", "--port=5003"]
