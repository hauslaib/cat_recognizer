ARG BUILD_FROM=ghcr.io/hassio-addons/base-python:3.9
FROM $BUILD_FROM

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/src/app

# Copy requirements and script
COPY requirements.txt ./
COPY cat_recognizer.py ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Run the cat recognizer script
CMD ["python3", "cat_recognizer.py"]
