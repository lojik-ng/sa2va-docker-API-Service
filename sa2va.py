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

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
start_time = None
MODEL_PATH = os.getenv('MODEL_PATH', 'ByteDance/Sa2VA-1B')  # Default to ByteDance/Sa2VA-1B if not specified

def load_model():
    global model, tokenizer, start_time
    print(f"Loading model from path: {MODEL_PATH}")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    start_time = time.time()

def base64_to_image(base64_string):
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image

@app.route('/')
def get_uptime():
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
    try:
        data = request.get_json()
        if not data or 'base64Image' not in data or 'prompt' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Convert base64 to image
        image = base64_to_image(data['base64Image'])
        text_prompts = data['prompt']

        # Process with model
        input_dict = {
            'image': image,
            'text': text_prompts,
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': tokenizer,
        }
        
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
    app.run(host='0.0.0.0', port=3303, threaded=True)