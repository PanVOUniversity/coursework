# Dockerfile for frame segmentation pipeline (GPU optimized)
# Build: docker build -t frame-seg:gpu .

FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies (excluding torch/torchvision - already in base image)
COPY requirements.txt .
RUN pip install -r requirements.txt


# Install Playwright and Chromium
RUN playwright install chromium && \
    playwright install-deps chromium && \
    playwright install-deps firefox

# Install Detectron2 from source (most reliable for GPU)
RUN git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2 && \
    cd /tmp/detectron2 && \
    pip install -e . && \
    cd /app && \
    rm -rf /tmp/detectron2

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright


