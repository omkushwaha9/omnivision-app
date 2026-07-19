# ============================================================
# OmniVision — Custom Dataset Loader
# Use this INSTEAD of the CIFAR-10 cells when training on
# your own images.
#
# Required folder structure:
#   dataset/
#   ├── train/
#   │   ├── class_a/  ← one subfolder per class
#   │   └── class_b/
#   └── val/
#       ├── class_a/
#       └── class_b/
#
# Upload your dataset to Google Drive, then run:
#   from google.colab import drive
#   drive.mount('/content/drive')
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── CONFIG — change these to match YOUR dataset ──────────────
DATASET_PATH = '/content/drive/MyDrive/my_dataset'  # update path
IMG_SIZE     = 96       # 96 or 128 recommended for transfer learning
BATCH_SIZE   = 32
EPOCHS_P1    = 15
EPOCHS_P2    = 10

# ── Load from folder structure ────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH + '/train',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH + '/val',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

# ── Augmentation via tf.data ──────────────────────────────────
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
], name="augmentation")

# Apply augmentation ONLY to training set
train_ds = train_ds.map(
    lambda x, y: (augment(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).cache().shuffle(1000).prefetch(AUTOTUNE)

val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ── Build MobileNetV2 model ───────────────────────────────────
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.30)(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.20)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)

cb = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint('best_model_custom.h5',
                              monitor='val_accuracy',
                              save_best_only=True, verbose=1),
]

# Phase 1 — frozen base
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(train_ds, epochs=EPOCHS_P1,
                    validation_data=val_ds, callbacks=cb)

# Phase 2 — unfreeze last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history_ft = model.fit(train_ds, epochs=EPOCHS_P2,
                       validation_data=val_ds, callbacks=cb)

# ── Evaluate ──────────────────────────────────────────────────
model_best = tf.keras.models.load_model('best_model_custom.h5')
loss, acc  = model_best.evaluate(val_ds, verbose=0)
print(f"\nVal Accuracy: {acc*100:.2f}%  |  Val Loss: {loss:.4f}")

# Predictions for confusion matrix
y_true, y_pred = [], []
for imgs, labels in val_ds:
    preds  = model_best.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(max(8, NUM_CLASSES), max(6, NUM_CLASSES-2)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — Custom Dataset')
plt.tight_layout()
plt.savefig('confusion_matrix_custom.png', dpi=150)
plt.show()

# ── Export ────────────────────────────────────────────────────
model_best.save('object_classifier_custom.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model_best)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('object_classifier_custom.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"TFLite: {len(tflite_model)/1e6:.2f} MB")

# Download
from google.colab import files
files.download('object_classifier_custom.h5')
files.download('object_classifier_custom.tflite')
print("Done — update CLASS_NAMES and IMG_SIZE in app.py to match")
print("CLASS_NAMES =", CLASS_NAMES)
print("IMG_SIZE    =", (IMG_SIZE, IMG_SIZE))
