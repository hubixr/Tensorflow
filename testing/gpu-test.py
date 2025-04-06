#!/usr/bin/env python3
"""
TensorFlow GPU Benchmarking Script
- Handles CUDA initialization properly
- Optimized for RTX 5070 (Compute Capability 12.0)
- Includes memory management
- Mixed precision & XLA support
- Benchmarking metrics
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================================
# 1. Environment Configuration
# ==============================================

# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise SystemError("No GPU detected!")

# Configure GPU
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
print(f"\nGPU Detected: {gpus[0].name.split('/')[-1]}")

# ==============================================
# 2. Model Definition
# ==============================================

def create_benchmark_model(input_shape=(224, 224, 3), num_classes=10):
    """Create a benchmark CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ==============================================
# 3. Benchmark Setup
# ==============================================

# Generate synthetic data
def generate_data(num_samples=1000):
    images = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
    labels = np.random.randint(0, 10, (num_samples,))
    return images, labels

# Benchmark function
def run_benchmark(model, x, y, epochs=5, batch_size=32):
    start_time = time.time()
    history = model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    duration = time.time() - start_time
    return duration / epochs  # Return time per epoch

# ==============================================
# 4. Main Execution
# ==============================================

def main():
    # Create and compile model
    model = create_benchmark_model()
    
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Enable XLA compilation
    tf.config.optimizer.set_jit(True)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate data
    train_images, train_labels = generate_data(2000)
    
    # Warm-up run
    print("\nRunning warm-up...")
    model.fit(train_images[:100], train_labels[:100], epochs=1, verbose=0)
    
    # Benchmark different batch sizes
    batch_sizes = [32, 64, 128, 256]
    results = {}
    
    print("\nStarting benchmarks...")
    for batch_size in batch_sizes:
        try:
            avg_time = run_benchmark(
                model, 
                train_images, 
                train_labels,
                epochs=3,
                batch_size=batch_size
            )
            results[batch_size] = avg_time
            print(f"Batch {batch_size:3d} | {avg_time:.3f} sec/epoch | ", end='')
            
            # Get memory stats
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"GPU Mem: {mem_info['peak']//(1024**2)}MB")
            
        except tf.errors.ResourceExhaustedError:
            print(f"Batch {batch_size:3d} | OOM Error - Reduce batch size")
            break
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    for bs, time_ in results.items():
        print(f"Batch {bs:3d}: {time_:.3f} sec/epoch | {2000/bs*time_:.2f} samples/sec")

if __name__ == "__main__":
    main()