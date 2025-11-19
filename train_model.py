import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil

print("TensorFlow version:", tf.__version__)

# -------------------------
# Parameters
# -------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4
DATASET_PATH = 'dataset'
LIMIT_PER_CLASS = 2500
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# -------------------------
# Check dataset structure
# -------------------------
def check_dataset_structure():
    expected_folders = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    possible_paths = [
        DATASET_PATH,
        os.path.join(DATASET_PATH, 'train'),
        os.path.join(DATASET_PATH, 'Alzheimer_s Dataset', 'train')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            found = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            if any(f in found for f in expected_folders):
                print(f"✅ Dataset found at: {path}")
                return path
    print("❌ Dataset not found! Check your folder structure.")
    exit(1)

# -------------------------
# Limit dataset to 2500/class
# -------------------------
def limit_dataset_samples(base_path, limit_per_class=2500, out_dir='limited_dataset'):
    classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    for cls in classes:
        src = os.path.join(base_path, cls)
        dst = os.path.join(out_dir, cls)
        os.makedirs(dst, exist_ok=True)
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        selected = random.sample(files, min(limit_per_class, len(files)))
        for f in selected:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        print(f"✅ {cls}: {len(selected)} images copied.")
    return out_dir

dataset_path = check_dataset_structure()
limited_path = limit_dataset_samples(dataset_path, LIMIT_PER_CLASS)

# -------------------------
# Data Generators
# -------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    limited_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=RANDOM_SEED
)

val_gen = datagen.flow_from_directory(
    limited_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=RANDOM_SEED
)

# -------------------------
# Fast Model: MobileNetV3Small
# -------------------------
def create_fast_model():
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # freeze for fast training

    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = create_fast_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# Callbacks (use .keras format)
# -------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('fast_best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# -------------------------
# Training
# -------------------------
print("🚀 Training fast model...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

model.save('fast_alzheimer_model.keras')
print("💾 Model saved as 'fast_alzheimer_model.keras'")

# -------------------------
# Plot training
# -------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('fast_training_history.png', dpi=300)
plt.show()

# -------------------------
# Evaluation
# -------------------------
print("📊 Evaluating best model...")
best_model = tf.keras.models.load_model('fast_best_model.keras')
loss, acc = best_model.evaluate(val_gen)
print(f"✅ Final Accuracy: {acc:.4f}, Loss: {loss:.4f}")

print("🎉 Fast training pipeline completed successfully!")
