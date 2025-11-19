# train_oasis_model_fixed.py
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("TensorFlow version:", tf.__version__)

# -------------------------
# Parameters
# -------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16  # Reduced for stability
EPOCHS = 20
NUM_CLASSES = 4
DATASET_PATH = 'dataset/oasis/processed_images'
RANDOM_SEED = 42

# -------------------------
# Enhanced Data Preparation
# -------------------------
def check_and_create_dataset():
    """Check if dataset exists and create if needed"""
    
    # Check if processed images exist and have data
    if os.path.exists(DATASET_PATH):
        classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        total_images = 0
        
        for cls in classes:
            class_path = os.path.join(DATASET_PATH, cls)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                total_images += len(images)
                logger.info(f"Found {len(images)} images in {cls}")
        
        if total_images > 0:
            logger.info(f"✅ Dataset ready with {total_images} total images")
            return DATASET_PATH
        else:
            logger.warning("❌ Dataset folder exists but contains no images")
    
    # If no dataset exists, try to create it
    logger.info("🔄 Attempting to create dataset...")
    try:
        from preprocess_oasis import OASISDataProcessor
        
        processor = OASISDataProcessor(
            cross_sectional_csv='dataset/oasis/cross-sectional.csv',
            longitudinal_csv='dataset/oasis/longitudinal.csv',
            raw_data_path='dataset/oasis/OAS2_RAW_part1',
            output_path=DATASET_PATH
        )
        
        if processor.setup_dataset():
            processor.merge_csv_files()
            processor.save_merged_csv('dataset/oasis/oasis_merged.csv')
            processed_count = processor.preprocess_nifti_to_images()
            
            if processed_count > 0:
                logger.info(f"✅ Dataset created with {processed_count} subjects")
                return DATASET_PATH
            else:
                logger.error("❌ Dataset creation failed - no images processed")
                return None
        else:
            logger.error("❌ Dataset setup failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Dataset creation error: {e}")
        return None

def create_synthetic_dataset():
    """Create a small synthetic dataset for testing if real data is not available"""
    logger.warning("🔄 Creating synthetic dataset for testing...")
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    # Create synthetic images
    for cls in classes:
        class_path = os.path.join(DATASET_PATH, cls)
        os.makedirs(class_path, exist_ok=True)
        
        # Create 100 synthetic images per class
        for i in range(100):
            # Create random brain-like images
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            
            # Add some structure to make it look like MRI
            center_x, center_y = 64, 64
            cv2.circle(img, (center_x, center_y), 30, (150, 150, 150), -1)
            cv2.circle(img, (center_x, center_y), 15, (200, 200, 200), -1)
            
            filename = f"synthetic_{cls}_{i:03d}.png"
            cv2.imwrite(os.path.join(class_path, filename), img)
        
        logger.info(f"Created 100 synthetic images for {cls}")
    
    logger.info("✅ Synthetic dataset created for testing")
    return DATASET_PATH

# -------------------------
# Robust Data Generators
# -------------------------
def create_data_generators(dataset_path):
    """Create data generators with extensive error handling"""
    
    # Verify dataset has images
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    has_images = False
    
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                has_images = True
                logger.info(f"✅ {cls}: {len(images)} images")
            else:
                logger.warning(f"⚠️ {cls}: No images found")
        else:
            logger.warning(f"⚠️ {cls}: Folder not found")
    
    if not has_images:
        logger.error("❌ No images found in any class!")
        return None, None
    
    # Create data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    try:
        train_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            seed=RANDOM_SEED,
            shuffle=True
        )
        
        val_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            seed=RANDOM_SEED,
            shuffle=False
        )
        
        logger.info(f"✅ Training samples: {train_gen.samples}")
        logger.info(f"✅ Validation samples: {val_gen.samples}")
        logger.info(f"✅ Class indices: {train_gen.class_indices}")
        
        if train_gen.samples == 0 or val_gen.samples == 0:
            logger.error("❌ No samples found in generators!")
            return None, None
            
        return train_gen, val_gen
        
    except Exception as e:
        logger.error(f"❌ Error creating data generators: {e}")
        return None, None

# -------------------------
# Simple Model
# -------------------------
def create_simple_model():
    """Create a simple CNN model for testing"""
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_mobilenet_model():
    """Create MobileNet model"""
    
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# -------------------------
# Training Function
# -------------------------
def train_oasis_model():
    """Main training function with comprehensive error handling"""
    
    logger.info("🚀 Starting OASIS Alzheimer's Detection Training")
    
    # Step 1: Prepare dataset
    logger.info("📊 Step 1: Preparing dataset...")
    dataset_path = check_and_create_dataset()
    
    if dataset_path is None:
        logger.warning("❌ Real dataset not available. Creating synthetic dataset for testing...")
        dataset_path = create_synthetic_dataset()
        if dataset_path is None:
            logger.error("❌ Could not create any dataset!")
            return None, None
    
    # Step 2: Create data generators
    logger.info("📊 Step 2: Creating data generators...")
    train_gen, val_gen = create_data_generators(dataset_path)
    
    if train_gen is None or val_gen is None:
        logger.error("❌ Failed to create data generators!")
        return None, None
    
    # Step 3: Create model
    logger.info("🧠 Step 3: Creating model...")
    try:
        model = create_mobilenet_model()
    except:
        logger.warning("⚠️ MobileNet failed, using simple CNN...")
        model = create_simple_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'oasis_best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Step 4: Training
    logger.info("🎯 Step 4: Starting training...")
    try:
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=min(100, train_gen.samples // BATCH_SIZE),  # Limit steps for testing
            validation_steps=min(50, val_gen.samples // BATCH_SIZE)
        )
        
        # Save final model
        model.save('oasis_alzheimer_model.keras')
        logger.info("💾 Model saved as 'oasis_alzheimer_model.keras'")
        
        return history, model
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None, None

# -------------------------
# Visualization
# -------------------------
def plot_training_history(history):
    """Plot training history"""
    if history is None:
        logger.warning("No history to plot")
        return
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('oasis_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    print("🏥 OASIS Alzheimer's Detection Training")
    print("=" * 50)
    
    # Train model
    history, model = train_oasis_model()
    
    if history is not None and model is not None:
        # Plot results
        plot_training_history(history)
        
        # Evaluate model
        print("📊 Evaluating model...")
        dataset_path = DATASET_PATH
        train_gen, val_gen = create_data_generators(dataset_path)
        
        if val_gen is not None:
            # Load best model for evaluation
            if os.path.exists('oasis_best_model.keras'):
                best_model = tf.keras.models.load_model('oasis_best_model.keras')
                results = best_model.evaluate(val_gen, verbose=1)
                print(f"✅ Final Evaluation - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
            else:
                results = model.evaluate(val_gen, verbose=1)
                print(f"✅ Final Evaluation - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")
        
        print("🎉 OASIS model training completed successfully!")
    else:
        print("❌ Training failed! Check the logs above for details.")