import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Parametry sieci
image_size = 128
wavelength = 2.14e-3  # w metrach (140 GHz)
propagation_distance = 0.7  # w metrach
pixel_size = 0.9e-3  # w metrach

# Funkcja generująca przykładowe dane treningowe i testowe
def generate_data(samples=1000):
    X = np.random.rand(samples, image_size, image_size, 2) * 2 - 1  # Losowe pola optyczne (część rzeczywista i urojona)
    Y = np.zeros((samples, image_size, image_size, 1))  # Zerowa macierz
    center = image_size // 2
    radius = 3  # Promień 3 piksele, czyli średnica 6 pikseli
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:
                Y[:, center + i, center + j, 0] = 1  # Intensywność w okręgu 6x6
    return X, Y

# Generowanie danych
X_train, Y_train = generate_data(800)
X_test, Y_test = generate_data(200)

# Normalizacja danych wyjściowych
Y_train /= np.max(Y_train)
Y_test /= np.max(Y_test)

# Wyświetlenie przykładowych danych
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.title('Wejście')
    
    plt.subplot(2, 5, i+6)
    plt.imshow(Y_train[i, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.title('Docelowa')
plt.show()

# Definicja warstwy propagacji THz
class THzPropagationLayer(Layer):
    def __init__(self, **kwargs):
        super(THzPropagationLayer, self).__init__(**kwargs)
        self.kernel = self.create_kernel()
    
    def create_kernel(self):
        k = 2 * np.pi / wavelength
        x = np.linspace(-image_size//2, image_size//2, image_size) * pixel_size
        X, Y = np.meshgrid(x, x)
        h = np.exp(1j * k / (2 * propagation_distance) * (X**2 + Y**2))
        h = np.fft.fftshift(h)
        h_real, h_imag = np.real(h), np.imag(h)
        h_complex = np.stack([h_real, h_imag], axis=-1)
        return tf.convert_to_tensor(h_complex, dtype=tf.float32)
    
    def call(self, inputs):
        u_real, u_imag = inputs[..., 0], inputs[..., 1]
        h_real, h_imag = self.kernel[..., 0], self.kernel[..., 1]
        
        u_real_fft = tf.signal.fft2d(tf.complex(u_real, u_imag))
        h_fft = tf.signal.fft2d(tf.complex(h_real, h_imag))
        
        output_fft = u_real_fft * h_fft
        output = tf.signal.ifft2d(output_fft)
        intensity = tf.abs(output)  # Konwersja na intensywność
        intensity /= tf.reduce_max(intensity)  # Normalizacja intensywności
        return tf.expand_dims(intensity, axis=-1)  # Dopasowanie wymiaru do Y_train

# Warstwa definiująca fazę DOE
class PhaseMaskLayer(Layer):
    def __init__(self, **kwargs):
        super(PhaseMaskLayer, self).__init__(**kwargs)
        self.phase = self.add_weight(name="phase", shape=(image_size, image_size),
                                     initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), trainable=True)
    
    def call(self, inputs):
        u_real, u_imag = inputs[..., 0], inputs[..., 1]
        u_complex = tf.complex(u_real, u_imag) * tf.exp(1j * tf.complex(self.phase, 0.0))
        return tf.stack([tf.math.real(u_complex), tf.math.imag(u_complex)], axis=-1)

# Definicja modelu
input_layer = Input(shape=(image_size, image_size, 2))
phase_layer = PhaseMaskLayer()(input_layer)
output_layer = THzPropagationLayer()(phase_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # Zmniejszono LR
              loss=tf.keras.losses.MeanAbsoluteError())  # Zmiana funkcji straty

model.summary()

# Trenowanie modelu
history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_test, Y_test))  # Więcej epok

# Ewaluacja modelu
test_loss = model.evaluate(X_test, Y_test)
print(f'Test Loss: {test_loss}')

# Wykres strat
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Loss (train)')
plt.plot(history.history['val_loss'], label='Loss (validation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Wykres strat w czasie treningu')
plt.show()