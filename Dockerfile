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
    # ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio einops timm pillow
RUN pip3 install git+https://github.com/huggingface/transformers
RUN pip3 install git+https://github.com/huggingface/accelerate
RUN pip3 install git+https://github.com/huggingface/diffusers
RUN pip3 install huggingface-hub
RUN pip3 install sentencepiece bitsandbytes protobuf decord numpy
RUN pip3 install flask
RUN pip3 install flask-cors
RUN pip install ninja==1.10.2.4 packaging setuptools
RUN ls && python3 -m pip install flash-attn --no-build-isolation
# RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Copy the application code
COPY sa2va.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV MODEL_PATH=ByteDance/Sa2VA-1B

# Create directory for model cache
RUN mkdir -p /app/model_cache

# Expose the port
EXPOSE 3303

# Run the application
CMD ["python3", "sa2va.py"]
