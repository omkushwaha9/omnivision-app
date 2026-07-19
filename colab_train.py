# ============================================================
# OmniVision — Google Colab Training Notebook (OPTIONAL)
# The Flask app (app.py) now uses pretrained ImageNet MobileNetV2
# out of the box — no training required for 1000 everyday objects.
# Use this notebook only if you want a custom CIFAR-10 fine-tuned model.
# Run each cell block in order in Google Colab
# Runtime → Change Runtime Type → GPU (T4)
# ============================================================

# ── CELL 1: Installs & Imports ───────────────────────────────
!pip install tensorflow matplotlib scikit-learn seaborn -q

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

print("TF Version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))


# ── CELL 2: Load & Preprocess CIFAR-10 ──────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = len(CLASS_NAMES)

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# One-hot encode labels
y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(f"Train: {x_train.shape}  Test: {x_test.shape}")


# ── CELL 3: Data Augmentation ────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Preview
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for imgs, _ in datagen.flow(x_train[:10], y_train_oh[:10], batch_size=10):
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i])
        ax.axis('off')
    break
plt.suptitle("Augmented Training Samples")
plt.tight_layout()
plt.show()


# ── CELL 4: MobileNetV2 Transfer Learning ───────────────────
IMG_SIZE_TL = 96
BATCH_SIZE  = 64

base_model = MobileNetV2(
    input_shape=(IMG_SIZE_TL, IMG_SIZE_TL, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = layers.Input(shape=(32, 32, 3))
x       = layers.Lambda(lambda img:
              tf.image.resize(img, [IMG_SIZE_TL, IMG_SIZE_TL]))(inputs)
x       = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x       = base_model(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.30)(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.20)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.summary()


# ── CELL 5: Compile & Train Phase 1 (Frozen Base) ───────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                              save_best_only=True, verbose=1),
]

history = model.fit(
    datagen.flow(x_train, y_train_oh, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=15,
    validation_data=(x_test, y_test_oh),
    callbacks=cb
)


# ── CELL 6: Fine-Tune Phase 2 (Unfreeze Last 30 Layers) ─────
print("\n=== Phase 2: Fine-tuning ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_ft = model.fit(
    datagen.flow(x_train, y_train_oh, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=10,
    validation_data=(x_test, y_test_oh),
    callbacks=cb
)


# ── CELL 7: Evaluate ────────────────────────────────────────
model_best = tf.keras.models.load_model('best_model.h5')
loss, acc  = model_best.evaluate(x_test, y_test_oh, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%  |  Test Loss: {loss:.4f}")

y_pred = np.argmax(model_best.predict(x_test), axis=1)
y_true = y_test.flatten()
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
all_acc   = history.history['accuracy']     + history_ft.history['accuracy']
all_val   = history.history['val_accuracy'] + history_ft.history['val_accuracy']
all_loss  = history.history['loss']         + history_ft.history['loss']
all_vloss = history.history['val_loss']     + history_ft.history['val_loss']
ax1.plot(all_acc, label='Train'); ax1.plot(all_val, label='Val')
ax1.set_title('Accuracy'); ax1.legend()
ax2.plot(all_loss, label='Train'); ax2.plot(all_vloss, label='Val')
ax2.set_title('Loss'); ax2.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()


# ── CELL 8: Export & Download ────────────────────────────────
# Export to TFLite (smaller + faster for deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model_best)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('object_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"TFLite: {len(tflite_model)/1e6:.2f} MB")

# Also save Keras .h5 for Flask server
model_best.save('object_classifier.h5')
print("Keras .h5 saved")

# Download to your computer
from google.colab import files
files.download('object_classifier.h5')
files.download('object_classifier.tflite')
files.download('confusion_matrix.png')
files.download('training_curves.png')
print("Done — check your Downloads folder")
