# summary.py

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.use("Agg")

# 📥 Ładowanie EMNIST (byclass)
print("Loading EMNIST...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = ds_info.features['label'].num_classes
INPUT_SHAPE = (28, 28, 1)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

ds_test = ds_test.map(preprocess).batch(256).prefetch(tf.data.AUTOTUNE)

x_test_np = []
y_test_np = []

for batch_x, batch_y in ds_test:
    x_test_np.append(batch_x.numpy())
    y_test_np.append(batch_y.numpy())

x_test = np.concatenate(x_test_np, axis=0)
y_test = np.concatenate(y_test_np, axis=0)

results = []
MODELS_DIR = 'models'
best_acc = 0
best_cm = None
best_model_name = None

for filename in os.listdir(MODELS_DIR):
    if filename.endswith(".h5"):
        model_path = os.path.join(MODELS_DIR, filename)
        print(f"\nEvaluating model: {filename}")
        model = tf.keras.models.load_model(model_path)

        y_pred_probs = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        results.append({
            'Model': filename,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        })

        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrix(y_test, y_pred)
            best_model_name = filename

df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print("\n=== Model Performance Summary ===")
print(df)

df.to_csv('emnist_model_comparison.csv', index=False)

# Confusion matrix – best model
plt.figure(figsize=(12, 10))
sns.heatmap(best_cm, cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix – Best Model: {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_best_model.png')
plt.close()

# Line plot – accuracy of all models
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Model', y='Accuracy', marker='o', sort=False, palette='viridis')
plt.title('Accuracy Comparison of Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')  # Rotate model names for better readability
plt.tight_layout()
plt.savefig('accuracy_comparison_lineplot.png')
plt.close()

# Barplot – accuracy of all models
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Accuracy', y='Model', palette='viridis')
plt.title('Accuracy porównanie modeli')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('accuracy_comparison_barplot.png')
plt.close()

print("\n✅ Wszystko gotowe! Wyniki zapisane:")
print("- emnist_model_comparison.csv")
print("- confusion_matrix_best_model.png")
print("- accuracy_comparison_barplot.png")
