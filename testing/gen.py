import tensorflow as tf
import numpy as np
import time

def create_benchmark_model(input_dim=1000, output_dim=10, layers=5, neurons=512):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation='relu', input_shape=(input_dim,)))
    for _ in range(layers - 1):
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_fake_data(samples=10000, input_dim=1000, output_dim=10):
    X = np.random.rand(samples, input_dim).astype(np.float32)
    y = np.random.randint(0, output_dim, size=(samples,))
    return X, y

def benchmark_model(model, X, y, epochs=100, batch_size=256):
    start_time = time.time()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Benchmark completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    print("TensorFlow CPU Benchmark Starting...")
    model = create_benchmark_model()
    X, y = generate_fake_data()
    benchmark_model(model, X, y)