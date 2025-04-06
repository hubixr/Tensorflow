#!/usr/bin/env python3
"""
Adaptive CNN Trainer - Automatically uses GPU if available, falls back to CPU
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import psutil  # For memory monitoring

# ==============================================
# 1. Configuration
# ==============================================

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress verbose logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Training parameters
BATCH_SIZE = 256
IMG_SIZE = 224
NUM_CLASSES = 1000
EPOCHS = 10
NUM_SAMPLES = 10000  # Reduced from original large value

# ==============================================
# 2. Device Detection & Configuration
# ==============================================

def configure_device():
    """Automatically configures for GPU or CPU"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable GPU optimizations
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision policy for GPU
            mixed_precision.set_global_policy('mixed_float16')
            tf.config.optimizer.set_jit(True)
            
            device_name = f"GPU: {gpus[0].name.split('/')[-1]}"
            print(f"\n⚡ GPU detected - Enabling acceleration")
            
            # Multi-GPU support if available
            if len(gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                print(f"Using {len(gpus)} GPUs with mirrored strategy")
                return strategy, device_name
                
            return tf.distribute.get_strategy(), device_name
            
        except RuntimeError as e:
            print(f"⚠️ GPU configuration failed: {e}")
    
    # Fallback to CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("\n🔧 No GPU available - Falling back to CPU")
    return tf.distribute.get_strategy(), "CPU"

# ==============================================
# 3. Model Definition
# ==============================================

def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Creates a GPU-optimized model that also works on CPU"""
    model = models.Sequential([
        # Input block
        layers.Conv2D(64, (7, 7), strides=2, padding='same', 
                     activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=2),
        
        # Convolutional blocks
        *[conv_block(filters) for filters in [64, 128, 256, 512]],
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model

def conv_block(filters):
    """Helper to create convolutional blocks"""
    return [
        layers.Conv2D(filters, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2))
    ]

# ==============================================
# 4. Training Utilities
# ==============================================

def check_memory_available(required_gb=2):
    """Check if enough memory is available"""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    if available_gb < required_gb:
        raise MemoryError(f"Only {available_gb:.1f}GB RAM available, need at least {required_gb}GB")

def generate_data(num_samples, img_size=IMG_SIZE, batch_size=1000):
    """Generates synthetic image data in batches to avoid memory issues"""
    check_memory_available()
    
    print(f"\n🧪 Generating {num_samples} samples in batches of {batch_size}...")
    
    # Calculate number of batches needed
    num_batches = (num_samples + batch_size - 1) // batch_size
    remaining_samples = num_samples
    
    x_data = []
    y_data = []
    
    for i in range(num_batches):
        current_batch = min(batch_size, remaining_samples)
        print(f"Generating batch {i+1}/{num_batches} ({current_batch} samples)")
        
        x_data.append(np.random.rand(current_batch, img_size, img_size, 3).astype(np.float32))
        y_data.append(np.random.randint(0, NUM_CLASSES, (current_batch,)))
        remaining_samples -= current_batch
        
        # Clear memory between batches
        if i % 5 == 0 and i > 0:
            tf.keras.backend.clear_session()
    
    # Concatenate all batches
    x_train = np.concatenate(x_data, axis=0)
    y_train = np.concatenate(y_data, axis=0)
    
    return x_train, y_train
    
def train_model(model, x, y, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """Trains model with timing and memory tracking"""
    # Warm-up
    print("\n🔥 Warming up...")
    model.fit(x[:batch_size], y[:batch_size], epochs=1, verbose=0)
    
    # Training
    print("\n🚀 Starting training...")
    start_time = time.time()
    history = model.fit(
        x, y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Metrics
    avg_epoch_time = training_time / epochs
    throughput = len(x) / (avg_epoch_time * batch_size)
    
    return avg_epoch_time, throughput, history

# ==============================================
# 5. Main Execution
# ==============================================

def main():
    try:
        # Device configuration
        strategy, device_name = configure_device()
        
        # Data generation
        x_train, y_train = generate_data(NUM_SAMPLES)
        
        with strategy.scope():
            # Model creation
            print("\n🛠️ Creating model...")
            model = create_model()
            
            # Compilation
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Display summary
        print("\n📊 Model Summary:")
        model.summary()
        
        # Training
        print(f"\n💻 Training on: {device_name}")
        avg_time, throughput, history = train_model(model, x_train, y_train)
        
        # Results
        print("\n📈 Training Results:")
        print(f"- Device: {device_name}")
        print(f"- Average epoch time: {avg_time:.2f} seconds")
        print(f"- Throughput: {throughput:.1f} samples/second")
        print(f"- Final accuracy: {history.history['accuracy'][-1]:.2f}")
        
        # Memory info
        if 'GPU' in device_name:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"- GPU memory used: {mem_info['peak']//(1024**2)}MB")
        
    except MemoryError as e:
        print(f"\n❌ Memory Error: {e}")
        print("Try reducing NUM_SAMPLES or IMG_SIZE in the configuration")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()