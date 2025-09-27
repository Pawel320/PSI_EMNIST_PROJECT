import tensorflow as tf
import tensorflow_datasets as tfds
#import matplotlib.pyplot as plt
import numpy as np
import random

# Wczytanie modelu
model = tf.keras.models.load_model("create_super_advanced_emnist_model.h5")
print("✅ Model załadowany.")

# EMNIST - ten sam preprocessing co wcześniej
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

# Ładowanie danych testowych
ds_test = tfds.load('emnist/byclass', split='test', as_supervised=True)
ds_test = ds_test.map(preprocess).batch(1)

# Klasy EMNIST byclass: 62 znaków (litery i cyfry)
# Dokumentacja: https://www.nist.gov/itl/products-and-services/emnist-dataset
# Tu uproszczona mapa etykiet (dla demonstracji)
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r',
    't', 'y', 'c', 'i', 'j', 'k', 'l', 'm', 'o', 'p',
    's', 'u', 'v', 'w', 'x', 'z'
]

# Wybierz losową próbkę
for image, label in ds_test.shuffle(1000).take(1):
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    img_np = image[0].numpy().squeeze()  # (28, 28)

    # Obrót i odbicie lustrzane (kolejność ważna!)
    img_fixed = np.rot90(np.fliplr(img_np), k=1)


    # plt.imshow(img_fixed, cmap="gray")
    # plt.title(f"Prawdziwa: {emnist_labels[int(label)]} / Przewidziana: {emnist_labels[predicted_class]}")
    # plt.axis("off")
    # plt.show()

# Ładowanie i przygotowanie zbioru treningowego
ds_train = tfds.load('emnist/byclass', split='train', as_supervised=True)
ds_train = ds_train.map(preprocess).batch(32)

# Oblicz accuracy na zbiorze treningowym
train_loss, train_acc = model.evaluate(ds_train, verbose=0)
print(f"✅ Accuracy na zbiorze treningowym: {train_acc:.4f}")

# Oblicz accuracy na zbiorze testowym
test_loss, test_acc = model.evaluate(ds_test, verbose=0)
print(f"✅ Accuracy na zbiorze testowym: {test_acc:.4f}")
