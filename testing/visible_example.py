import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parametry fizyczne
wavelength = 1e-3       # długość fali [m]
k = 2 * np.pi / wavelength  # liczba falowa
dx = 4e-6                   # rozmiar neuronu / pitch piksela [m]
d = 0.05                    # odległość między warstwami [m]

# Wczytanie danych
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizacja i rozmiar 80x80 (Twoje warstwy)
x_train = tf.image.resize(tf.expand_dims(x_train, -1), (80, 80)) / 255.0
x_test = tf.image.resize(tf.expand_dims(x_test, -1), (80, 80)) / 255.0

# Konwersja do tf.complex64 z zerową fazą
x_train = tf.cast(tf.squeeze(x_train), tf.complex64)
x_test = tf.cast(tf.squeeze(x_test), tf.complex64)

# Dodaj batch dimension
x_train = tf.expand_dims(x_train, axis=-1)  # [batch, 80, 80, 1]
x_test = tf.expand_dims(x_test, axis=-1)

# Jedna faza = pojedynczy kanał (wycinamy 3D)
x_train = tf.squeeze(x_train, axis=-1)
x_test = tf.squeeze(x_test, axis=-1)

# One-hot encoding etykiet
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

class ClassificationD2NN(tf.keras.Model):
    def __init__(self, num_layers, layer_shape, d, wavelength, dx, num_classes):
        super(ClassificationD2NN, self).__init__()
        self.d2nn = D2NNModel(num_layers, layer_shape, d, wavelength, dx)
        self.num_classes = num_classes

        # Tu: uczymy się maski przypisującej obszary detekcji klasom
        self.detector_mask = self.add_weight(
            shape=(num_classes, *layer_shape),  # [10, 80, 80]
            initializer='random_uniform',
            trainable=False,  # lub True, jeśli chcesz współuczyć
            name="detector_mask"
        )

    def call(self, inputs):
        field = self.d2nn(inputs)
        intensity = tf.abs(field)**2  # [batch, 80, 80]
        
        # Liczymy sumy energii dla każdej klasy (przez maskę)
        logits = tf.reduce_sum(
            tf.multiply(
                tf.expand_dims(intensity, axis=1),  # [batch, 1, 80, 80]
                tf.expand_dims(self.detector_mask, axis=0)  # [1, 10, 80, 80]
            ),
            axis=[2, 3]  # sumujemy po przestrzeni
        )
        return logits


# Funkcja obliczająca jądro propagacyjne (angular spectrum)
def propagation_kernel(shape, d, wavelength, dx):
    """
    Oblicza jądro propagacyjne H dla propagacji metodą angular spectrum.
    shape: (H, W) – rozmiar obrazu
    d: odległość propagacji [m]
    wavelength: długość fali [m]
    dx: rozmiar piksela [m]
    """
    H, W = shape
    # Częstotliwości przestrzenne
    fx = tf.linspace(-1/(2*dx), 1/(2*dx), W)
    fy = tf.linspace(-1/(2*dx), 1/(2*dx), H)
    FX, FY = tf.meshgrid(fx, fy)
    # Argument pod pierwiastkiem musi być >= 0
    arg = 1 - (wavelength**2) * (FX**2 + FY**2)
    # Używamy tf.where, aby unikać ujemnych wartości (można przyjąć, że tam jądro = 0)
    arg = tf.where(arg < 0, tf.zeros_like(arg), arg)
    k = 2 * np.pi / wavelength  # liczba falowa
    phase = k * d * tf.sqrt(arg)
    H_kernel = tf.exp(tf.complex(tf.zeros_like(phase), phase))
    return H_kernel

# Funkcja propagacji pola za pomocą FFT (angular spectrum)
def propagate(field, d, wavelength, dx):
    """
    Propaguje pole optyczne 'field' na odległość d.
    """
    # Obliczenie jądra propagacyjnego
    H_kernel = propagation_kernel(field.shape[1:3], d, wavelength, dx)
    # FFT pola (zakładamy, że pole jest typu tf.complex64)
    field_ft = tf.signal.fft2d(field)
    # Mnożenie w dziedzinie Fouriera
    field_ft_prop = field_ft * tf.cast(H_kernel, field_ft.dtype)
    # Powrót do dziedziny przestrzennej
    field_propagated = tf.signal.ifft2d(field_ft_prop)
    return field_propagated

# Warstwa dyfrakcyjna – modulacja fazowa
class DiffractiveLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        """
        shape: tuple (H, W) określający liczbę "neuronów" w warstwie.
        """
        super(DiffractiveLayer, self).__init__(name=name)
        self.shape_ = shape

    def build(self, input_shape):
        # Inicjalizujemy fazę z małymi losowymi wartościami
        phase_init = tf.random.uniform(self.shape_, minval=0, maxval=2*np.pi)
        # Trenowalna zmienna – faza (dla trybu phase-only amplituda = 1)
        self.phase = self.add_weight(
            shape=self.shape_,
            initializer=tf.keras.initializers.Constant(phase_init),
            trainable=True,
            name="phase"
        )
        super(DiffractiveLayer, self).build(input_shape)

    def call(self, inputs):

        # Można dodatkowo ograniczyć zakres fazy (np. poprzez sigmoid, jeśli wymagane)
        # Tworzenie transmitancji: exp(i * faza)
        complex_phase = tf.complex(tf.zeros_like(self.phase), self.phase)
        t = tf.exp(complex_phase)
        # Mnożenie element-po-element: modulacja fazowa pola wejściowego
        return inputs * tf.cast(t, inputs.dtype)

# Model D2NN – łańcuch warstw dyfrakcyjnych i propagacji między nimi
class D2NNModel(tf.keras.Model):
    def __init__(self, num_layers, layer_shape, d, wavelength, dx):
        """
        num_layers: liczba warstw dyfrakcyjnych
        layer_shape: (H, W) – rozmiar każdej warstwy
        d: odległość między kolejnymi warstwami [m]
        wavelength: długość fali [m]
        dx: rozmiar neuronu/piksela [m]
        """
        super(D2NNModel, self).__init__()
        self.num_layers = num_layers
        self.d = d
        self.wavelength = wavelength
        self.dx = dx
        # Tworzymy listę warstw dyfrakcyjnych
        self.diff_layers = [DiffractiveLayer(layer_shape, name=f"diff_layer_{i+1}") 
                            for i in range(num_layers)]
        
    def call(self, inputs):
        """
        inputs: pole wejściowe jako tensor tf.complex64 o kształcie [batch, H, W]
        """
        field = inputs
        # Przechodzimy przez wszystkie warstwy – na przemian dyfrakcja + propagacja
        for layer in self.diff_layers:
            # Modulacja fazowa
            field = layer(field)
            # Propagacja do następnej warstwy
            field = propagate(field, self.d, self.wavelength, self.dx)
        return field

# Przykładowe użycie modelu
if __name__ == '__main__':
    # Ustal rozmiar obrazu / warstwy
    H, W = 80, 80  # liczba neuronów w poziomie i pionie (przykładowo)
    batch_size = 1

    # Stwórz przykładowe wejście – np. amplitudowe przedstawienie obiektu (tutaj: okrąg)
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    radius = 0.5
    mask = np.where(X**2 + Y**2 <= radius**2, 1.0, 0.0)
    # Wstępnie zakodowane pole – amplituda = mask, faza = 0
    input_field = tf.cast(mask, tf.complex64)

    # Dodaj wymiar batcha
    input_field = tf.expand_dims(input_field, axis=0)  # kształt [1, H, W]

    # Utwórz model D2NN z przykładowo 5 warstwami
    num_layers = 5
    model = D2NNModel(num_layers=num_layers, layer_shape=(H, W), d=d, wavelength=wavelength, dx=dx)

    # Przeprowadź propagację wejścia przez D2NN
    output_field = model(input_field)

    # Aby uzyskać „detekcję” – np. intensywność pola w ostatniej warstwie
    intensity = tf.abs(output_field)**2

# Parametry
num_classes = 10
model = ClassificationD2NN(num_layers=5, layer_shape=(80, 80), d=d,
                           wavelength=wavelength, dx=dx, num_classes=num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Dataset
batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Trening
history = model.fit(train_ds, epochs=2, validation_data=test_ds)

pd.DataFrame(history.history).plot(figsize=(10,7), xlabel = "epochs")
plt.xlabel("epochs")
plt.savefig("loss.png")
