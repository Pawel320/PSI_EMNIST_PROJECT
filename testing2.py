import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')


emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

img = Image.open('test.jpg').convert('L').resize((28, 28))
img_np = np.array(img)
img_fixed = np.fliplr(np.rot90(img_np, k=3))
img_input = img_fixed.reshape(1, 28, 28, 1).astype('float32') / 255.0

model = tf.keras.models.load_model("emnist_advanced_updated.h5")
pred = model.predict(img_input)
pred_idx = np.argmax(pred)
pred_char = emnist_labels[pred_idx]
confidence = float(np.max(pred)) * 100

plt.imshow(img_np, cmap='gray')
plt.title(f'Predykcja: {pred_char} (klasa {pred_idx})\nPewność: {confidence:.1f}%')
plt.axis('off')
plt.show()