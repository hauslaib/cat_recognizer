ARG BUILD_FROM
FROM $BUILD_FROM

# Install build dependencies
RUN apk add --no-cache python3 python3-dev py3-pip build-base jpeg-dev zlib-dev openblas-dev freetype-dev libpng-dev

# Set the working directory
WORKDIR /usr/src/app

# Copy requirements and script
COPY requirements.txt ./
COPY cat_recognizer.py ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Run the cat recognizer script
CMD ["python3", "cat_recognizer.py"]