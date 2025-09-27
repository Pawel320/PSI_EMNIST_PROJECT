
# main_resnet_train.py
import tensorflow as tf
import tensorflow_datasets as tfds

# Parametry
BATCH_SIZE = 128
EPOCHS = 10

MODEL_PATH = "emnist_model_updated4.h5"               # <-- Model wejściowy
OUTPUT_MODEL_PATH = "emnist_model_updated5.h5"  # <-- Model po kontynuacji

# Wczytanie modelu
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model załadowany.")

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


# Wczytywanie danych
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)
NUM_CLASSES = ds_info.features['label'].num_classes

# Przygotowanie zbioru treningowego
ds_train_orig = ds_train.map(preprocess)
ds_train_aug = ds_train_orig.map(augment)
ds_train = ds_train_orig.concatenate(ds_train_aug)
ds_train = ds_train.shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Przygotowanie zbioru testowego
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Tworzenie i kompilacja modelu
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Trenowanie
print("🚀 Trening modelu ResNet-like na EMNIST...")
model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)

# Zapis modelu
model.save(MODEL_PATH)
print(f"💾 Model zapisany jako {MODEL_PATH}")

