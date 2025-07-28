import sys
import os
import io
import time
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set environment variables for better performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

print("DEBUG: Script started")

try:
    from flask import Flask, request, send_file, jsonify, render_template
    from flask_cors import CORS
    from PIL import Image
    import numpy as np
    import torch
    import gdown
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print("DEBUG: All imports successful")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configuration
MODEL_SCALE = 4
WEIGHTS_FOLDER = 'weights'
MODEL_NAME = 'RealESRGAN_x4plus'
WEIGHTS_FILENAME = f'{MODEL_NAME}.pth'
WEIGHTS_PATH = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILENAME)

# Processing parameters
MAX_IMAGE_SIZE = 1024  # Reduced for Render's memory limits
TILE_SIZE = 256
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit for Render
MAX_WORKERS = min(4, os.cpu_count() or 1)

# Create weights directory
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

# Model URLs - using the correct RealESRGAN model
MODEL_URLS = {
    'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
}

def download_model():
    """Download the correct RealESRGAN model"""
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading {WEIGHTS_FILENAME}...")
        try:
            # Use direct download from GitHub releases
            import urllib.request
            urllib.request.urlretrieve(MODEL_URLS[MODEL_NAME], WEIGHTS_PATH)
            print(f"Downloaded {WEIGHTS_FILENAME} successfully")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    return True

def setup_model():
    """Setup RealESRGAN model with correct architecture"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Define the correct model architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Create RealESRGANer instance
        upsampler = RealESRGANer(
            scale=4,
            model_path=WEIGHTS_PATH,
            model=model,
            tile=TILE_SIZE,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),  # Use half precision on GPU
            device=device
        )
        
        print("Model loaded successfully")
        return upsampler
    except Exception as e:
        print(f"Error setting up model: {e}")
        return None

# Global model instance
MODEL_INSTANCE = None
MODEL_LOCK = threading.Lock()

def get_model():
    """Get or create model instance (thread-safe)"""
    global MODEL_INSTANCE
    
    if MODEL_INSTANCE is None:
        with MODEL_LOCK:
            if MODEL_INSTANCE is None:
                MODEL_INSTANCE = setup_model()
    
    return MODEL_INSTANCE

def smart_resize(img, max_size=MAX_IMAGE_SIZE):
    """Resize image if too large"""
    width, height = img.size
    
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        print(f"Resizing from {img.size} to ({new_width}, {new_height})")
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img

# Download model on startup
print("Downloading model...")
if not download_model():
    print("Failed to download model")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Warm up model
print("Loading model...")
try:
    model = get_model()
    if model is None:
        raise Exception("Failed to load model")
    print("Model loaded and ready!")
except Exception as e:
    print(f"Model initialization failed: {e}")

@app.route('/')
def index():
    """Serve the main page"""
    try:
        return render_template('index.html')
    except:
        return jsonify({
            'message': 'AI Image Enhancer API',
            'endpoints': {
                'enhance': '/api/enhance (POST)',
                'status': '/api/status (GET)',
                'health': '/health (GET)'
            }
        })

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    """Main image enhancement endpoint"""
    print("Enhancement request received")
    
    # Check if model is loaded
    model = get_model()
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE//1024//1024}MB'}), 400
    
    try:
        # Load and preprocess image
        start_time = time.time()
        img = Image.open(file.stream).convert('RGB')
        print(f"Original size: {img.size}")
        
        # Resize if necessary
        img = smart_resize(img)
        print(f"Processing size: {img.size}")
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Enhance image
        print("Starting enhancement...")
        try:
            output, _ = model.enhance(img_array, outscale=4)
            print(f"Enhancement completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Enhancement error: {e}")
            return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500
        
        # Convert back to PIL Image
        enhanced_img = Image.fromarray(output)
        print(f"Enhanced size: {enhanced_img.size}")
        
        # Save to memory buffer
        img_buffer = io.BytesIO()
        
        # Choose format based on image size
        if enhanced_img.size[0] * enhanced_img.size[1] > 2000000:
            enhanced_img.save(img_buffer, 'JPEG', quality=90, optimize=True)
            mimetype = 'image/jpeg'
        else:
            enhanced_img.save(img_buffer, 'PNG', optimize=True)
            mimetype = 'image/png'
        
        img_buffer.seek(0)
        
        # Cleanup
        del img_array, output, enhanced_img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return send_file(img_buffer, mimetype=mimetype)
    
    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get API status"""
    model = get_model()
    
    status = {
        'status': 'healthy' if model is not None else 'model_error',
        'model_loaded': model is not None,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'cuda_available': torch.cuda.is_available(),
        'max_image_size': MAX_IMAGE_SIZE,
        'max_file_size_mb': MAX_FILE_SIZE // 1024 // 1024,
        'model_name': MODEL_NAME,
        'scale': MODEL_SCALE
    }
    
    if torch.cuda.is_available():
        try:
            status['gpu_memory'] = {
                'total_mb': torch.cuda.get_device_properties(0).total_memory // 1024 // 1024,
                'allocated_mb': torch.cuda.memory_allocated() // 1024 // 1024,
                'cached_mb': torch.cuda.memory_reserved() // 1024 // 1024
            }
        except:
            pass
    
    return jsonify(status)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model = get_model()
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'timestamp': time.time()
    })

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({'message': 'API is working!', 'version': '2.0'})

if __name__ == '__main__':
    print("Starting Flask app...")
    port = int(os.environ.get("PORT", 10000))
    
    # Use Gunicorn in production, Flask dev server locally
    if os.environ.get('RENDER'):
        # This won't be reached when using Gunicorn, but kept for reference
        app.run(host="0.0.0.0", port=port, threaded=True)
    else:
        app.run(host="0.0.0.0", port=port, debug=True)
