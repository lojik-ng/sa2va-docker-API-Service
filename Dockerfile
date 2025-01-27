FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV MODEL_PATH=ByteDance/Sa2VA-1B
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Upgrade pip and install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch and related packages
RUN pip3 install --no-cache-dir torch torchvision torchaudio einops timm pillow

# Install Hugging Face packages
RUN pip3 install --no-cache-dir \
    git+https://github.com/huggingface/transformers \
    git+https://github.com/huggingface/accelerate \
    git+https://github.com/huggingface/diffusers \
    huggingface-hub

# Install other dependencies
RUN pip3 install --no-cache-dir \
    sentencepiece \
    bitsandbytes \
    protobuf \
    decord \
    numpy \
    flask \
    flask-cors \
    peft \
    ninja==1.10.2.4 \
    packaging

# Install flash-attention
RUN python3 -m pip install --no-cache-dir flash-attn --no-build-isolation

# Create directory for model cache
RUN mkdir -p /app/model_cache

# Copy the application code
COPY sa2va.py .

# Expose the port
EXPOSE 3303

# Run the application with Python unbuffered output
CMD ["python3", "-u", "sa2va.py"]
