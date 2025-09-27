# trening_mnist.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Parametry
BATCH_SIZE = 128
EPOCHS = 10

MODEL_PATH = "emnist_model3_updated.h5"               # <-- Model wejściowy
OUTPUT_MODEL_PATH = "emnist_model3_updated_Mnist.h5"        # <-- Model po kontynuacji

# Wczytanie wcześniej wytrenowanego modelu
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model załadowany.")

# Preprocessing
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # Dodaj kanał dla CNN
    return image, label  # Etykiety 0-9 pasują do EMNIST

# Ładowanie MNIST
(ds_train, ds_test), info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Przetwarzanie i batchowanie danych
ds_train = ds_train.map(preprocess).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Kompilacja modelu (upewniamy się, że jest gotowy do treningu)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trenowanie modelu na MNIST
print("🔁 Trenowanie na MNIST...")
model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)

# Zapis zaktualizowanego modelu
model.save(OUTPUT_MODEL_PATH)
print(f"💾 Zapisano model z treningiem MNIST jako {OUTPUT_MODEL_PATH}")
