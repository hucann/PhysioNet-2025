# Use Python 3.13 base image
FROM python:3.13.2-slim

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
## Optional: install system dependencies (e.g. for wfdb/h5py)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libhdf5-dev \
        && rm -rf /var/lib/apt/lists/*

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt