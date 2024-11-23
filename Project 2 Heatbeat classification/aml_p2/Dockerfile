# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install Python and required libraries
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /

# Copy the requirements.txt file to the container
COPY requirements.txt requirements.txt

# Install the required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Set the command to run your application
CMD ["python3.10", "main.py"]

