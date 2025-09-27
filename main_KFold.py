# main_KFold.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Parametry
BATCH_SIZE = 128
EPOCHS = 32
NUM_FOLDS = 5
MODEL_BASENAME = "emnist_model_KFold"

# Augmentacja
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1, 0.1),
])

# Preprocessing i augmentacja
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

def augment(image, label):
    image = augmentation(image)
    return image, label

# Ładowanie danych jako numpy arrays
(ds_train, _), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
    batch_size=-1
)

images, labels = tfds.as_numpy(ds_train)
images = (images.astype(np.float32) / 255.0)[..., np.newaxis]
labels = labels.astype(np.int32)

NUM_CLASSES = ds_info.features['label'].num_classes

def create_best_emnist_model(input_shape=(28, 28, 1), num_classes=62):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# K-Fold cross-validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
    print(f"\n📂 Fold {fold + 1}/{NUM_FOLDS}")

    # Podział danych
    x_train, y_train = images[train_idx], labels[train_idx]
    x_val, y_val = images[val_idx], labels[val_idx]

    # Konwersja do tf.data
    ds_train_fold = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_val_fold = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Mapowanie augmentacji i batchowanie
    ds_train_fold = ds_train_fold.map(lambda x, y: (augmentation(x), y))
    ds_train_fold = ds_train_fold.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_val_fold = ds_val_fold.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Tworzenie modelu
    model = create_best_emnist_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Trenowanie
    model.fit(ds_train_fold, epochs=EPOCHS, validation_data=ds_val_fold)

    # Zapis modelu
    model_path = f"{MODEL_BASENAME}_{fold+1}.h5"
    model.save(model_path)
    print(f"💾 Zapisano model fold {fold+1}: {model_path}")