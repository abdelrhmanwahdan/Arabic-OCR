# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and clean up afterwards to reduce image size
RUN apt-get update && apt-get install -y \
    libgirepository1.0-dev \
    build-essential \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    gir1.2-gtk-3.0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements files into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
