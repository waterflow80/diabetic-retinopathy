FROM python:3.11-slim

RUN apt-get update

WORKDIR /drdetection

RUN pip install --no-cache-dir --upgrade pip

RUN mkdir -p /drdetection/plots /drdetection/data /drdetection/models

# Copy Train and Test data
COPY data/train_images_part /drdetection/data/train_images_part
COPY data/test_images_part /drdetection/data/test_images_part
COPY data/train.csv /drdetection/data/train.csv
COPY data/test.csv /drdetection/data/test.csv

# Copy Source Code
COPY src/ /drdetection/src
COPY pyproject.toml /drdetection/pyproject.toml

# Copy Entrypoint script
COPY train_test.sh /drdetection/train_test.sh

# Install Dependencies
RUN pip install -e .

# Train and Test Model
CMD /bin/bash train_test.sh


