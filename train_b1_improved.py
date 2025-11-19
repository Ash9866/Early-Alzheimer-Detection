# train_b1_improved.py
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# --- CONFIG ---
DATA_DIR = "dataset"   # layout: dataset/<class_name>/*.jpg
MODEL_SAVE_PATH = "app/deep_learning/models/efficientnet_b1_improved.h5"
IMG_SIZE = (240, 240)        # EfficientNet-B1 commonly uses 240x240.
BATCH_SIZE = 16              # if GPU is available; reduce to 8-16 for CPU
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
VALIDATION_SPLIT = 0.2

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- DATA ---
print("Loading dataset from:", DATA_DIR)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=SEED
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# -------------------------------
# compute class weights by scanning directories
# -------------------------------
def count_files_per_class(data_dir, class_names):
    counts = []
    p = Path(data_dir)
    for cls in class_names:
        cls_dir = p / cls
        if not cls_dir.exists():
            counts.append(0)
            continue
        cnt = sum(1 for _ in cls_dir.rglob("*") if _.is_file() and not _.name.startswith('.'))
        counts.append(cnt)
    return counts

total_counts = count_files_per_class(DATA_DIR, class_names)
print("Total files per class (all data):", dict(zip(class_names, total_counts)))

# Estimate training counts after validation split (approximation)
train_counts = [int(math.floor(c * (1.0 - VALIDATION_SPLIT))) for c in total_counts]
print("Estimated training files per class:", dict(zip(class_names, train_counts)))

y_train_labels = np.concatenate([
    np.full(count, idx, dtype=np.int32) if count > 0 else np.array([], dtype=np.int32)
    for idx, count in enumerate(train_counts)
]) if sum(train_counts) > 0 else np.array([], dtype=np.int32)

if y_train_labels.size == 0:
    # fallback: compute weights from total counts
    print("No training split labels found (edge case). Falling back to total counts for class weights.")
    y_train_labels = np.concatenate([
        np.full(c, idx, dtype=np.int32) if c > 0 else np.array([], dtype=np.int32)
        for idx, c in enumerate(total_counts)
    ])

if y_train_labels.size == 0:
    class_weights = {i: 1.0 for i in range(num_classes)}
else:
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_train_labels
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_arr)}

print("Class weights:", class_weights)

# PREFETCH / CACHING
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# DATA AUGMENTATION
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05),
], name="data_augmentation")

# Build model
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)

# ---- FIX: use a Keras layer to convert uint8->float and scale ----
x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)
# optionally, if you want EfficientNet-specific preprocessing, wrap it as a Lambda:
# from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preproc
# x = layers.Lambda(lambda t: eff_preproc(t), name="eff_preprocess")(x)

base_model = tf.keras.applications.EfficientNetB1(
    include_top=False,
    weights='imagenet',
    input_tensor=x,
    pooling=None
)
base_model.trainable = False  # freeze for initial training

y = layers.GlobalAveragePooling2D()(base_model.output)
y = layers.BatchNormalization()(y)
y = layers.Dropout(0.4)(y)
y = layers.Dense(256, activation='relu')(y)
y = layers.Dropout(0.3)(y)
outputs = layers.Dense(num_classes, activation='softmax')(y)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# CALLBACKS
callbacks = [
    keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
]

# TRAIN HEAD
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# UNFREEZE top layers of base_model for fine-tuning
base_model.trainable = True

# Option: only unfreeze last N layers (safer). Here we unfreeze the top 40 layers:
for layer in base_model.layers[:-40]:
    layer.trainable = False

# recompile with lower LR
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

initial_epoch = history1.epoch[-1] + 1 if getattr(history1, "epoch", None) else 0
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    class_weight=class_weights
)

# Final evaluation
loss, acc = model.evaluate(val_ds)
print(f"Final validation accuracy: {acc:.4f}")

model.save(MODEL_SAVE_PATH)
print("Saved model to", MODEL_SAVE_PATH)
