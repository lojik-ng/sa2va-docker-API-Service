# Sa2VA API Service

This project provides a Flask-based REST API service that utilizes ByteDance's Sa2VA models for dense grounded understanding of images. Sa2VA (Segment Anything 2 Visual Assistant) is a powerful multimodal language model that combines SAM2 with LLaVA capabilities, enabling advanced image analysis, question answering, and dense object segmentation.

## Features

- **Model Capabilities**: 
  - Question answering about images
  - Optical Character Recognition
  - Hand writing recognition
  - Visual prompt understanding
  - Dense object segmentation
  - State-of-the-art performance on image grounding and segmentation benchmarks

- **API Features**:
  - RESTful API endpoints
  - Base64 image input support
  - Custom prompt configuration
  - GPU-accelerated inference
  - Service health monitoring
  - Docker containerization

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lojik-ng/sa2va-docker-API-Service.git
cd sa2va-docker-API-Service
```

2. Build the Docker image:
```bash
docker build -t sa2va-service .
```

3. Run the container:
```bash
docker run --gpus all -p 3303:3303 sa2va-service
```

### Environment Variables

You can customize the service using the following environment variables:

- `MODEL_PATH`: Path to the model (default: "ByteDance/Sa2VA-1B")
- `TRANSFORMERS_CACHE`: Cache directory for model files (default: "/app/model_cache")

Example with custom model path:
```bash
docker run --gpus all -p 3303:3303 -e MODEL_PATH=ByteDance/Sa2VA-1B --name sa2va sa2va
```

### Models:

- ByteDance/Sa2VA-1B
- ByteDance/Sa2VA-4B
- ByteDance/Sa2VA-8B
- ByteDance/Sa2VA-26B

## API Endpoints

### 1. Root Endpoint
- **URL**: `GET /`
- **Description**: Welcome endpoint providing available API endpoints
- **Response**:
```json
{
    "message": "Welcome to the Sa2VA API Server",
    "endpoints": {
        "/health": "GET - Health check and server status",
        "/process": "POST - Process image with text prompt"
    }
}
```

### 2. Health Check
- **URL**: `GET /health`
- **Description**: Service health status and configuration
- **Response**:
```json
{
    "status": "running",
    "uptime": "0:10:30",
    "started_at": "2025-01-22T23:41:56",
    "model_path": "ByteDance/Sa2VA-1B",
    "endpoints": {
        "/health": "GET - Health check and server status",
        "/process": "POST - Process image with text prompt"
    }
}
```

### 3. Image Processing
- **URL**: `POST /process`
- **Description**: Process an image with the Sa2VA model
- **Headers**: 
  - Content-Type: application/json
- **Request Body**:
```json
{
    "base64Image": "base64_encoded_image_string",
    "prompt": "your_analysis_prompt"
}
```
- **Response**:
```json
{
    "result": "Model's response to your prompt"
}
```

## Testing

Here's an example of how to test the API using curl:

```bash
#!/bin/bash

# Convert image to base64
BASE64_IMAGE=$(base64 -w 0 your_image.jpg)

# Create JSON payload file
cat > payload.json << EOF
{
  "base64Image": "${BASE64_IMAGE}",
  "prompt": "Analyze this image and describe what you see."
}
EOF

# Send request using the JSON file
curl -X POST \
  -H "Content-Type: application/json" \
  -d @payload.json \
  http://localhost:3303/process

# Clean up
rm payload.json
```

## Model Details

Sa2VA is built on InternVL2.5 and offers:
- Comparable performance to state-of-the-art MLLMs like Qwen2-VL and InternVL2.5
- Enhanced capabilities in visual prompt understanding
- Superior performance in dense object segmentation
- Support for both image and video analysis

For more information about the model, visit [Sa2VA github](https://github.com/magic-research/Sa2VA).

## License

This project uses the Sa2VA model which is subject to ByteDance's license terms. Please refer to the model's [license information](https://github.com/magic-research/Sa2VA) for usage terms and conditions.
