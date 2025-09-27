import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

model = tf.keras.models.load_model('emnist_advanced.h5')
img = Image.open('test2.jpg').convert('L')
img_np = np.array(img)

height, width = img_np.shape
assert height == 28, "Wysokość obrazu musi wynosić 28 pikseli"
k = width // 28

chars_for_model = []
chars_for_plot = []
for i in range(k):
    char_img = img_np[:, i*28:(i+1)*28]
    chars_for_plot.append(char_img)  # oryginalny do wyświetlania
    char_img_rot = np.fliplr(np.rot90(char_img, k=3))  # do modelu
    chars_for_model.append(char_img_rot)

chars_np = np.array(chars_for_model)[..., np.newaxis] / 255.0

preds = model.predict(chars_np)
pred_labels = np.argmax(preds, axis=1)
pred_chars = [EMNIST_LABELS[i] for i in pred_labels]

fig, axs = plt.subplots(1, k, figsize=(2*k, 2))
for i, (char_img, label) in enumerate(zip(chars_for_plot, pred_chars)):
    confidence = np.max(preds[i])
    axs[i].imshow(char_img, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f'{label}\n{confidence:.2f}', fontsize=16)
plt.suptitle('Pojedyncze znaki i predykcje', y=1.15)
plt.tight_layout()
plt.show()