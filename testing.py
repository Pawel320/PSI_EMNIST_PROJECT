import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for Matplotlib

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Rest of your code remains unchanged

# Wczytanie modelu
model = tf.keras.models.load_model("emnist_advanced_updated.h5")
print("✅ Model załadowany.")

# EMNIST - ten sam preprocessing co wcześniej
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

# Ładowanie danych testowych
ds_test = tfds.load('emnist/byclass', split='test', as_supervised=True)
ds_test = ds_test.map(preprocess).batch(1)

# Mapowanie etykiet EMNIST byclass (62 klasy: 0-9, A-Z, a-z)
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
def test_emnist_model_tfds(model, ds, emnist_labels):
    user_input = input("Podaj kilka znaków (np. abc123): ")
    images_to_show = []
    titles = []

    for char in user_input:
        if char not in emnist_labels:
            print(f"Znak '{char}' nie występuje w EMNIST.")
            continue
        label_idx = emnist_labels.index(char)
        found = False
        for image, label in tfds.as_numpy(ds):
            if label == label_idx:
                found = True
                img_input = image.reshape(1, 28, 28, 1).astype('float32')
                pred = model.predict(img_input, verbose=0)
                pred_idx = np.argmax(pred)
                pred_char = emnist_labels[pred_idx]
                confidence = float(np.max(pred)) * 100
                img_fixed = np.fliplr(np.rot90(image.squeeze(), k=3))
                images_to_show.append(img_fixed)
                titles.append(f"'{char}'\nPred: '{pred_char}'\nPewność: {confidence:.1f}%")
                break
        if not found:
            print(f"Nie znaleziono przykładu dla znaku '{char}'.")

    if images_to_show:
        num_images = len(images_to_show)
        cols = 5  # Number of columns in the grid
        rows = (num_images + cols - 1) // cols  # Calculate rows dynamically

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 3))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, (img, title) in enumerate(zip(images_to_show, titles)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title, fontsize=8)
            axes[i].axis('off')

        # Hide any unused subplots
        for i in range(len(images_to_show), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

model = tf.keras.models.load_model('emnist_model_updated5.h5')
test_emnist_model_tfds(model, ds_test, emnist_labels)