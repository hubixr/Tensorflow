import tensorflow as tf
import time

# Check if TensorFlow sees the GPU
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.experimental.list_physical_devices('GPU'))

# Matrix multiplication test
shape = (5000, 5000)
a = tf.random.normal(shape)
b = tf.random.normal(shape)

# Time execution on GPU
with tf.device('/GPU:0'):
    print("Running on GPU...")
    start_time = time.time()
    c = tf.matmul(a, b)
    tf.print("Matrix multiplication (GPU) done.")
    print("Execution time (GPU):", time.time() - start_time, "seconds")

# Time execution on CPU for comparison
with tf.device('/CPU:0'):
    print("Running on CPU...")
    start_time = time.time()
    c = tf.matmul(a, b)
    tf.print("Matrix multiplication (CPU) done.")
    print("Execution time (CPU):", time.time() - start_time, "seconds")