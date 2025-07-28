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
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    print("DEBUG: Imported concurrent.futures")
    import threading
    print("DEBUG: Imported threading")
    import time
    print("DEBUG: Imported time")
    import queue
    print("DEBUG: Imported queue")
except Exception as e:
    print(f"Import error: {e}")
    exit(1)

print("DEBUG: All imports successful")

import os
import gdown

# Performance optimizations
MODEL_SCALE = 4  # Use 2x instead of 4x for faster processing
WEIGHTS_FOLDER = 'weights'
WEIGHTS_FILENAME = f'RealESRGAN_x{MODEL_SCALE}.pth'
WEIGHTS_PATH = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILENAME)

# Create weights dir if not exists
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

# Download model if not already present
model_url = f"https://drive.google.com/uc?id=1sydrWWe8VF1oR1sXb904UWFd6Tp4mvKF"

if not os.path.exists(WEIGHTS_PATH):
    print(f"Downloading {WEIGHTS_FILENAME}...")
    gdown.download(model_url, WEIGHTS_PATH, quiet=False)
    
# Optimized processing parameters
MAX_IMAGE_SIZE = 2048  # Increased for better quality/speed balance
TILE_SIZE = 256  # Smaller tiles for better memory usage and parallelization
OVERLAP = 32  # Overlap for seamless tile blending
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB limit

# Enhanced concurrent processing
MAX_WORKERS = min(8, (os.cpu_count() or 1) * 2)  # More aggressive threading
BATCH_SIZE = 4  # Process multiple tiles in batches

# Global model cache and memory management
MODEL_CACHE = {}
MEMORY_THRESHOLD = 0.8  # GPU memory threshold

def setup_torch_optimizations():
    """Setup PyTorch optimizations for maximum performance"""
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
    else:
        # CPU optimizations
        torch.set_num_threads(min(4, os.cpu_count() or 1))

def get_optimized_model():
    """Get or create an optimized model instance with caching"""
    thread_id = threading.get_ident()
    
    if thread_id not in MODEL_CACHE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=MODEL_SCALE)
        model.load_weights(WEIGHTS_PATH)
        model.model.eval()
        
        # Model optimizations
        if hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                model.model = torch.compile(model.model, mode='max-autotune')
                print(f"DEBUG: Model compiled for thread {thread_id}")
            except:
                print(f"DEBUG: Compilation failed for thread {thread_id}")
        
        # Enable mixed precision if available
        if device.type == 'cuda':
            try:
                model.model = model.model.half()
                print(f"DEBUG: Enabled FP16 for thread {thread_id}")
            except:
                print(f"DEBUG: FP16 not supported for thread {thread_id}")
        
        MODEL_CACHE[thread_id] = model
        print(f"DEBUG: Created optimized model for thread {thread_id}")
    
    return MODEL_CACHE[thread_id]

def smart_resize(img, max_size=MAX_IMAGE_SIZE):
    """Intelligent resizing with quality preservation"""
    width, height = img.size
    pixels = width * height
    
    # Only resize if significantly larger than max_size
    if max(width, height) > max_size * 1.5:
        # Calculate optimal size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        # Use high-quality resampling
        print(f"DEBUG: Smart resize from {img.size} to ({new_width}, {new_height})")
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img

def create_overlapping_tiles(img_array, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Create overlapping tiles for seamless reconstruction"""
    h, w, c = img_array.shape
    
    tiles = []
    positions = []
    
    stride = tile_size - overlap
    
    # Calculate tile positions
    y_positions = list(range(0, h - tile_size + 1, stride))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    
    x_positions = list(range(0, w - tile_size + 1, stride))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)
    
    # Extract tiles
    for y in y_positions:
        for x in x_positions:
            tile = img_array[y:y+tile_size, x:x+tile_size].copy()
            tiles.append(tile)
            positions.append((y, x))
    
    return tiles, positions

def blend_tiles(output, enhanced_tile, y, x, tile_size, overlap, scale):
    """Blend tiles with overlap for seamless reconstruction"""
    output_y = y * scale
    output_x = x * scale
    output_tile_size = tile_size * scale
    output_overlap = overlap * scale
    
    # Simple replacement for now - can be enhanced with alpha blending
    output[output_y:output_y+output_tile_size, 
           output_x:output_x+output_tile_size] = enhanced_tile

def process_tile_batch(args):
    """Process a batch of tiles together for better efficiency"""
    batch_tiles, batch_positions, batch_id = args
    
    try:
        model = get_optimized_model()
        enhanced_tiles = []
        
        # Process tiles in the batch
        with torch.no_grad():
            if torch.cuda.is_available():
                # Use autocast for mixed precision
                with torch.cuda.amp.autocast():
                    for tile in batch_tiles:
                        enhanced_tile = model.predict(tile)
                        enhanced_tiles.append(enhanced_tile)
            else:
                for tile in batch_tiles:
                    enhanced_tile = model.predict(tile)
                    enhanced_tiles.append(enhanced_tile)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"DEBUG: Processed batch {batch_id} with {len(batch_tiles)} tiles")
        return batch_id, enhanced_tiles, batch_positions
        
    except Exception as e:
        print(f"DEBUG: Error processing batch {batch_id}: {e}")
        return batch_id, None, batch_positions

def process_with_batched_tiles(img_array, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Enhanced tile processing with batching and overlap"""
    h, w, c = img_array.shape
    
    # Check if small enough to process directly
    if h <= tile_size and w <= tile_size:
        model = get_optimized_model()
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    return model.predict(img_array)
            else:
                return model.predict(img_array)
    
    print(f"DEBUG: Processing {h}x{w} image with batched overlapping tiles")
    
    # Create overlapping tiles
    tiles, positions = create_overlapping_tiles(img_array, tile_size, overlap)
    total_tiles = len(tiles)
    
    # Group tiles into batches
    tile_batches = []
    position_batches = []
    
    for i in range(0, total_tiles, BATCH_SIZE):
        batch_tiles = tiles[i:i+BATCH_SIZE]
        batch_positions = positions[i:i+BATCH_SIZE]
        tile_batches.append(batch_tiles)
        position_batches.append(batch_positions)
    
    # Initialize output
    scale = MODEL_SCALE
    output = np.zeros((h * scale, w * scale, c), dtype=np.uint8)
    
    # Process batches concurrently
    start_time = time.time()
    completed_batches = 0
    total_batches = len(tile_batches)
    
    # Prepare batch tasks
    batch_tasks = [(tile_batches[i], position_batches[i], i) 
                   for i in range(total_batches)]
    
    # Use ThreadPoolExecutor for I/O bound with some CPU work
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(process_tile_batch, task): task 
                          for task in batch_tasks}
        
        for future in as_completed(future_to_batch):
            try:
                batch_id, enhanced_tiles, batch_positions = future.result()
                completed_batches += 1
                
                if enhanced_tiles is not None:
                    # Reconstruct with blending
                    for enhanced_tile, (y, x) in zip(enhanced_tiles, batch_positions):
                        blend_tiles(output, enhanced_tile, y, x, tile_size, overlap, scale)
                
                # Progress tracking
                progress = (completed_batches / total_batches) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / completed_batches) * (total_batches - completed_batches) if completed_batches > 0 else 0
                
                print(f"DEBUG: Batch progress: {completed_batches}/{total_batches} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                
            except Exception as e:
                print(f"DEBUG: Error in batch processing: {e}")
    
    total_time = time.time() - start_time
    print(f"DEBUG: Batched processing completed in {total_time:.2f}s")
    
    # Cleanup
    del tiles, positions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output

def check_memory_usage():
    """Monitor memory usage and trigger cleanup if needed"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        if memory_allocated > MEMORY_THRESHOLD:
            torch.cuda.empty_cache()
            gc.collect()
            return True
    return False

# Initialize optimizations
setup_torch_optimizations()

# Create weights dir and load model
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

if not os.path.exists(WEIGHTS_PATH):
    print(f"Downloading {WEIGHTS_FILENAME}...")
    # Add download logic here
    raise FileNotFoundError(f"Missing weights at {WEIGHTS_PATH}. Please download manually.")

print("DEBUG: Setting up Flask app...")

app = Flask(__name__)
CORS(app, origins="*")

# Warm up the model
print("DEBUG: Warming up model...")
try:
    dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    _ = get_optimized_model()
    print("DEBUG: Model warmed up successfully!")
except Exception as e:
    print(f"DEBUG: Model warmup failed: {e}")

@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    print("DEBUG: /api/enhance called")
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB'}), 400
    
    try:
        # Fast image loading and preprocessing
        img = Image.open(file.stream).convert('RGB')
        print(f"DEBUG: Original size: {img.size}")
        
        # Smart resizing
        img = smart_resize(img)
        print(f"DEBUG: Processing size: {img.size}")
        
    except Exception as e:
        print(f"DEBUG: Invalid image: {e}")
        return jsonify({'error': 'Invalid image format.'}), 400
    
    try:
        print("DEBUG: Starting optimized enhancement...")
        start_time = time.time()
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Memory check before processing
        check_memory_usage()
        
        # Process with optimized batched tiles
        if max(img_array.shape[:2]) > TILE_SIZE:
            sr_img_array = process_with_batched_tiles(img_array, TILE_SIZE, OVERLAP)
        else:
            model = get_optimized_model()
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        sr_img_array = model.predict(img_array)
                else:
                    sr_img_array = model.predict(img_array)
        
        processing_time = time.time() - start_time
        print(f"DEBUG: Enhancement completed in {processing_time:.2f}s")
        
        # Convert result
        if isinstance(sr_img_array, np.ndarray):
            sr_img = Image.fromarray(sr_img_array)
        else:
            sr_img = sr_img_array
        
        print(f"DEBUG: Enhanced size: {sr_img.size}")
        
        # Cleanup
        del img_array, sr_img_array
        check_memory_usage()
        
    except Exception as e:
        print(f"DEBUG: Enhancement failed: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500
    
    try:
        # Optimized image saving
        byte_io = io.BytesIO()
        
        # Smart format selection
        pixels = sr_img.size[0] * sr_img.size[1]
        if pixels > 4000000:  # Very large images
            sr_img.save(byte_io, 'JPEG', quality=92, optimize=True, progressive=True)
            mimetype = 'image/jpeg'
        else:
            sr_img.save(byte_io, 'PNG', optimize=True, compress_level=6)
            mimetype = 'image/png'
        
        byte_io.seek(0)
        print("DEBUG: Image saved to buffer")
        
        # Final cleanup
        del sr_img
        gc.collect()
        
        return send_file(byte_io, mimetype=mimetype)
        
    except Exception as e:
        print(f"DEBUG: Error saving image: {e}")
        return jsonify({'error': f'Error saving enhanced image: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Enhanced status with performance metrics"""
    status = {
        'status': 'healthy',
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'model_scale': MODEL_SCALE,
        'max_image_size': MAX_IMAGE_SIZE,
        'tile_size': TILE_SIZE,
        'overlap': OVERLAP,
        'batch_size': BATCH_SIZE,
        'max_workers': MAX_WORKERS,
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'active_models': len(MODEL_CACHE),
        'optimizations': {
            'torch_compile': hasattr(torch, 'compile'),
            'mixed_precision': torch.cuda.is_available(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark if torch.cuda.is_available() else False
        }
    }
    
    if torch.cuda.is_available():
        status['cuda_memory'] = {
            'total': torch.cuda.get_device_properties(0).total_memory,
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved()
        }
    
    return jsonify(status)

@app.route('/test')
def test():
    return jsonify({'status': 'Flask is working!', 'message': 'Optimized version ready'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Optimized API is running'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("DEBUG: Starting optimized Flask app")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
