# model_definitions.py
import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 128
EPOCHS = 10
MODEL_PATH = "emnist_model_res.h5"

# Ładowanie danych
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)
NUM_CLASSES = ds_info.features['label'].num_classes

# Augmentacja
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1, 0.1),
])

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

def augment(image, label):
    image = data_augmentation(image)
    return image, label

# Oryginalny zbiór treningowy
ds_train_orig = ds_train.map(preprocess)

# Zaugmentowany zbiór treningowy
ds_train_aug = ds_train_orig.map(augment)

# Połączenie obu zbiorów
ds_train = ds_train_orig.concatenate(ds_train_aug)
ds_train = ds_train.shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # jeśli zmieniamy rozmiar przestrzenny lub liczbę filtrów, dostosowujemy shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def create_resnet_like_model(input_shape=(28, 28, 1), num_classes=62):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

model = create_resnet_like_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("🚀 Rozpoczynanie trenowania z augmentacją...")
model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)
model.save(MODEL_PATH)
print(f"💾 Model zapisany jako {MODEL_PATH}")