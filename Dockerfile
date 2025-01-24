FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages one by one
RUN pip3 install packaging
RUN pip3 install ninja
RUN pip3 install torch
RUN pip3 install torchvision
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
RUN pip3 -m install flash-attn --no-build-isolation

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
