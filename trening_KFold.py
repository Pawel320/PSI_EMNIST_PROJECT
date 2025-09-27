# main_kfold.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Parametry
BATCH_SIZE = 128
EPOCHS = 10
FOLDS = 5

MODEL_PATH = "emnist_model_updated5_aug3.h5"               # <-- Model wejściowy
OUTPUT_MODEL_PATH = "emnist_model_updated5_aug3_KFold.h5"  # <-- Model po kontynuacji

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model załadowany.")


# Ładowanie danych jako numpy
(ds_train,ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
    batch_size=-1  # pełny zbiór jako numpy
)
images_np, labels_np = tfds.as_numpy(ds_train)
images_np = images_np.astype(np.float32) / 255.0
images_np = np.expand_dims(images_np, -1)

NUM_CLASSES = ds_info.features['label'].num_classes

# Augmentacja
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1, 0.1),
])

def augment(image, label):
    return data_augmentation(image), label

# K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(images_np, labels_np)):
    print(f"\n==============================")
    print(f"🚀 Fold {fold + 1}/{FOLDS}")
    print(f"==============================")

    x_train, y_train = images_np[train_idx], labels_np[train_idx]
    x_val, y_val = images_np[val_idx], labels_np[val_idx]

    # Tworzenie zbiorów tf.data
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.map(augment).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



    # Kompilacja
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trenowanie
    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)

    # Zapis modelu
    model_path = OUTPUT_MODEL_PATH.format(fold + 1)
    model.save(OUTPUT_MODEL_PATH)
    print(f"✅ Zapisano model: {model_path}")
