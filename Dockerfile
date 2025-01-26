FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV MODEL_PATH=ByteDance/Sa2VA-1B

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# RUN pip3 install --upgrade pip
# Install Python packages with CUDA support
RUN pip3 install packaging
RUN pip3 install ninja
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install numpy
RUN pip3 install sympy
RUN pip3 install filelock
RUN pip3 install jinja2
RUN pip3 install networkx
RUN pip3 install einops
RUN pip3 install timm
RUN pip3 install triton
RUN pip3 install transformers
RUN pip3 install peft
RUN pip3 install Pillow
RUN pip3 install flask
RUN pip3 install flask-cors
RUN pip3 install flash-attn --no-build-isolation

# Copy the application code
COPY sa2va.py .

# Create directory for model cache
RUN mkdir -p /app/model_cache

# Expose the port
EXPOSE 3303

# Run the application
CMD ["python3", "sa2va.py"]
