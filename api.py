import sys
sys.stdout.flush()
import faulthandler; faulthandler.enable()
print("DEBUG: Script started")

try:
    from flask import Flask, request, send_file, jsonify, render_template
    print("DEBUG: Imported Flask")
    from flask_cors import CORS
    print("DEBUG: Imported flask_cors")
    from PIL import Image
    print("DEBUG: Imported PIL")
    from RealESRGAN import RealESRGAN
    print("DEBUG: Imported RealESRGAN")
    import numpy as np
    print("DEBUG: Imported numpy")
    import torch
    print("DEBUG: Imported torch")
    import io
    print("DEBUG: Imported io")
    import os
    print("DEBUG: Imported os")
    import gc
    print("DEBUG: Imported gc")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print("DEBUG: Imported concurrent.futures")
    import threading
    print("DEBUG: Imported threading")
    import time
    print("DEBUG: Imported time")
except Exception as e:
    print(f"Import error: {e}")
    exit(1)

print("DEBUG: All imports successful")

# Performance optimizations
MODEL_SCALE = 2  # Use 2x instead of 4x for faster processing
WEIGHTS_FOLDER = 'weights'
WEIGHTS_FILENAME = f'RealESRGAN_x{MODEL_SCALE}.pth'
WEIGHTS_PATH = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILENAME)

# Image processing limits
MAX_IMAGE_SIZE = 1024  # Limit input image size
TILE_SIZE = 512  # Process in tiles for large images
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

# Concurrent processing settings
MAX_WORKERS = min(4, os.cpu_count() or 1)  # Limit concurrent workers
THREAD_LOCAL = threading.local()  # For thread-local model instances

try:
    os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
    print("DEBUG: Created weights folder")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing weights at {WEIGHTS_PATH}. Please download manually.")
except Exception as e:
    print(f"Error preparing weights: {e}")
    exit(1)

print("DEBUG: Loading model ...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEBUG: Using device: {device}")
    
    # Performance optimizations for PyTorch
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        # CPU optimizations
        torch.set_num_threads(2)  # Reduce per-thread cores for concurrent processing
    
    # Create main model instance
    model = RealESRGAN(device, scale=MODEL_SCALE)
    model.load_weights(WEIGHTS_PATH)
    model.model.eval()
    torch.set_grad_enabled(False)
    
    print("DEBUG: Model loaded and optimized!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def get_thread_model():
    """Get or create a model instance for the current thread"""
    if not hasattr(THREAD_LOCAL, 'model'):
        # Create a new model instance for this thread
        thread_model = RealESRGAN(device, scale=MODEL_SCALE)
        thread_model.load_weights(WEIGHTS_PATH)
        thread_model.model.eval()
        THREAD_LOCAL.model = thread_model
        print(f"DEBUG: Created model for thread {threading.current_thread().name}")
    return THREAD_LOCAL.model

app = Flask(__name__)
CORS(app, origins="*")

def resize_if_too_large(img, max_size=MAX_IMAGE_SIZE):
    """Resize image if it's too large while maintaining aspect ratio"""
    width, height = img.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        print(f"DEBUG: Resizing from {img.size} to ({new_width}, {new_height})")
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

def process_tile(args):
    """Process a single tile - designed for concurrent execution"""
    tile_data, tile_info, tile_id = args
    
    try:
        # Get thread-local model instance
        thread_model = get_thread_model()
        
        # Process the tile
        with torch.no_grad():
            enhanced_tile = thread_model.predict(tile_data)
        
        # Clean up thread-local memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"DEBUG: Processed tile {tile_id}")
        return tile_id, enhanced_tile, tile_info
        
    except Exception as e:
        print(f"DEBUG: Error processing tile {tile_id}: {e}")
        return tile_id, None, tile_info

def process_in_tiles_concurrent(img_array, tile_size=TILE_SIZE):
    """Process large images in tiles using concurrent processing"""
    h, w, c = img_array.shape
    
    if h <= tile_size and w <= tile_size:
        # Small enough to process normally
        with torch.no_grad():
            return model.predict(img_array)
    
    print(f"DEBUG: Processing {h}x{w} image in tiles of {tile_size}x{tile_size} with {MAX_WORKERS} workers")
    
    # Calculate number of tiles
    h_tiles = (h + tile_size - 1) // tile_size
    w_tiles = (w + tile_size - 1) // tile_size
    total_tiles = h_tiles * w_tiles
    
    # Initialize output array
    scale = MODEL_SCALE
    output = np.zeros((h * scale, w * scale, c), dtype=np.uint8)
    
    # Prepare tile tasks
    tile_tasks = []
    tile_id = 0
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            # Calculate tile boundaries
            y_start = i * tile_size
            y_end = min((i + 1) * tile_size, h)
            x_start = j * tile_size
            x_end = min((j + 1) * tile_size, w)
            
            # Extract tile
            tile = img_array[y_start:y_end, x_start:x_end].copy()
            
            # Store tile info for reconstruction
            tile_info = {
                'output_y_start': y_start * scale,
                'output_y_end': y_end * scale,
                'output_x_start': x_start * scale,
                'output_x_end': x_end * scale
            }
            
            tile_tasks.append((tile, tile_info, tile_id))
            tile_id += 1
    
    # Process tiles concurrently
    start_time = time.time()
    completed_tiles = 0
    
    # Use ThreadPoolExecutor for CPU-bound tasks with I/O
    # For pure CPU tasks, ProcessPoolExecutor might be better but has more overhead
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_tile = {executor.submit(process_tile, task): task for task in tile_tasks}
        
        # Collect results as they complete
        for future in as_completed(future_to_tile):
            try:
                tile_id, enhanced_tile, tile_info = future.result()
                completed_tiles += 1
                
                if enhanced_tile is not None:
                    # Place enhanced tile in output
                    output[tile_info['output_y_start']:tile_info['output_y_end'], 
                           tile_info['output_x_start']:tile_info['output_x_end']] = enhanced_tile
                else:
                    print(f"DEBUG: Failed to process tile {tile_id}")
                
                # Progress update
                progress = (completed_tiles / total_tiles) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / completed_tiles) * (total_tiles - completed_tiles) if completed_tiles > 0 else 0
                
                print(f"DEBUG: Progress: {completed_tiles}/{total_tiles} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                
            except Exception as e:
                print(f"DEBUG: Error collecting tile result: {e}")
    
    total_time = time.time() - start_time
    print(f"DEBUG: Concurrent tile processing completed in {total_time:.2f}s")
    
    # Clean up memory
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return output

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    print("DEBUG: /api/enhance called")
    
    if 'image' not in request.files:
        print("DEBUG: No image provided")
        return jsonify({'error': 'No image provided.'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        print("DEBUG: Empty filename")
        return jsonify({'error': 'No image selected.'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB'}), 400
    
    try:
        img = Image.open(file.stream).convert('RGB')
        print(f"DEBUG: Image opened - Original size: {img.size}")
        
        # Resize if too large
        img = resize_if_too_large(img)
        print(f"DEBUG: Processing size: {img.size}")
        
    except Exception as e:
        print(f"DEBUG: Invalid image: {e}")
        return jsonify({'error': 'Invalid image format.'}), 400
    
    try:
        print("DEBUG: Starting model prediction...")
        img_array = np.array(img)
        print(f"DEBUG: Input array shape: {img_array.shape}")
        
        # Process image with concurrent tiling
        processing_start = time.time()
        
        if max(img_array.shape[:2]) > TILE_SIZE:
            sr_img_array = process_in_tiles_concurrent(img_array, TILE_SIZE)
        else:
            with torch.no_grad():
                sr_img_array = model.predict(img_array)
        
        processing_time = time.time() - processing_start
        print(f"DEBUG: Model prediction completed in {processing_time:.2f}s")
        
        # Convert to PIL Image
        if isinstance(sr_img_array, np.ndarray):
            sr_img = Image.fromarray(sr_img_array)
            print("DEBUG: Converted ndarray to Image")
        else:
            sr_img = sr_img_array
        
        print(f"DEBUG: Enhanced image size: {sr_img.size}")
        
        # Clean up memory
        del img_array, sr_img_array
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
            
    except Exception as e:
        print(f"DEBUG: Enhancement failed: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500
    
    try:
        byte_io = io.BytesIO()
        # Use JPEG for smaller file size if quality is acceptable
        if sr_img.size[0] * sr_img.size[1] > 2000000:  # Large images
            sr_img.save(byte_io, 'JPEG', quality=95, optimize=True)
            mimetype = 'image/jpeg'
        else:
            sr_img.save(byte_io, 'PNG', optimize=True)
            mimetype = 'image/png'
        
        byte_io.seek(0)
        print("DEBUG: Image saved to buffer")
        
        # Clean up
        del sr_img
        gc.collect()
        
        return send_file(byte_io, mimetype=mimetype)
    except Exception as e:
        print(f"DEBUG: Error saving image: {e}")
        return jsonify({'error': f'Error saving enhanced image: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_scale': MODEL_SCALE,
        'max_image_size': MAX_IMAGE_SIZE,
        'tile_size': TILE_SIZE,
        'max_workers': MAX_WORKERS,
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    })

@app.route('/test')
def test():
    return jsonify({'status': 'Flask is working!', 'message': 'Test endpoint successful'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("DEBUG: Starting Flask app on port 5000")
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port) # Disable debug mode for production