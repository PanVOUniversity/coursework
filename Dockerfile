# Dockerfile for frame segmentation pipeline
# Supports both CPU and GPU (CUDA) versions

# Base image - use CUDA version for GPU support
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

# Set working directory
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
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (required for Detectron2)
ARG CUDA=0
RUN if [ "$CUDA" = "1" ]; then \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install Python dependencies (excluding torch/torchvision as they're installed above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Install Playwright and Chromium
RUN playwright install chromium
RUN playwright install-deps chromium

# Install Detectron2
# For GPU version, use: docker build --build-arg CUDA=1 -t frame-seg .
# For CPU version, use: docker build -t frame-seg .
ARG CUDA=0
RUN if [ "$CUDA" = "1" ]; then \
        pip install 'git+https://github.com/facebookresearch/detectron2.git'; \
    else \
        pip install 'git+https://github.com/facebookresearch/detectron2.git' || \
        pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html || \
        pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.13/index.html; \
    fi

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Default command
CMD ["python", "scripts/html_generator.py", "--n", "10"]

