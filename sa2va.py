"""
Sa2VA API Server - ByteDance's Sa2VA-1B Vision-Language Model Interface

This Flask application provides a REST API interface to ByteDance's Sa2VA-1B model,
which is capable of processing images and text prompts to generate natural language responses.
The server handles image uploads in base64 format and returns model-generated responses.

Dependencies:
    - torch: For PyTorch deep learning operations
    - transformers: For loading and running the Sa2VA model
    - PIL: For image processing
    - flask: For web server functionality
    - flask_cors: For handling Cross-Origin Resource Sharing
"""

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from datetime import datetime, timedelta

# Initialize Flask application with CORS support
app = Flask(__name__)
CORS(app)

# Global variables for maintaining model state
model = None  # The Sa2VA model instance
tokenizer = None  # Tokenizer for processing text inputs
start_time = None  # Server start timestamp
MODEL_PATH = os.getenv('MODEL_PATH', 'ByteDance/Sa2VA-1B')  # Model path with default fallback

def load_model():
    """
    Initialize the Sa2VA model and tokenizer.
    
    This function loads the model with optimized settings:
    - Uses bfloat16 for reduced memory usage
    - Enables Flash Attention for faster processing
    - Moves model to CUDA for GPU acceleration
    """
    global model, tokenizer, start_time
    print(f"Loading model from path: {MODEL_PATH}")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        low_cpu_mem_usage=True,      # Optimize CPU memory usage
        use_flash_attn=True,         # Enable Flash Attention for faster processing
        trust_remote_code=True).eval().cuda()  # Move to GPU and set to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    start_time = time.time()

def base64_to_image(base64_string):
    """
    Convert a base64 string to a PIL Image object.
    
    Args:
        base64_string (str): The base64-encoded image string, with or without data URL prefix
        
    Returns:
        PIL.Image: The decoded image in RGB format
    """
    # Remove data URL prefix if present (e.g., 'data:image/jpeg;base64,')
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image

@app.route('/')
def get_uptime():
    """
    Health check endpoint that provides server status and configuration information.
    
    Returns:
        JSON response containing:
        - Server running status
        - Uptime duration
        - Server start time
        - Model path
        - API endpoint details
    """
    if start_time is None:
        return jsonify({'error': 'Service not fully initialized yet'})
    
    uptime_seconds = time.time() - start_time
    uptime = str(timedelta(seconds=int(uptime_seconds)))
    return jsonify({
        'status': 'running',
        'uptime': uptime,
        'started_at': datetime.fromtimestamp(start_time).isoformat(),
        'model': MODEL_PATH,
        'endpoint': '/api',
        'method': 'POST',
        'params': 'base64Image, prompt'
    })

@app.route('/api', methods=['POST'])
def process_image():
    """
    Main API endpoint for processing images with the Sa2VA model.
    
    Expected JSON payload:
        - base64Image: Base64-encoded image data
        - prompt: Text prompt for the model
        
    Returns:
        JSON response containing either:
        - result: Model's generated response
        - error: Error message if processing failed
    """
    try:
        # Validate input data
        data = request.get_json()
        if not data or 'base64Image' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Process input data
        image = base64_to_image(data['base64Image'])
        text_prompts = data['prompt']

        # Prepare model inputs
        input_dict = {
            'image': image,
            'text': text_prompts,
            'past_text': '',          # No conversation history
            'mask_prompts': None,     # No masking applied
            'tokenizer': tokenizer,
        }
        
        # Generate prediction using the model
        with torch.inference_mode():
            return_dict = model.predict_forward(**input_dict)
            answer = return_dict["prediction"]

        return jsonify({'result': answer})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model into GPU...")
    load_model()
    print("Model loaded successfully! Starting server...")
    # Start Flask server with threading enabled for handling multiple requests
    app.run(host='0.0.0.0', port=3303, threaded=True)