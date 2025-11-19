import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
from model import create_mobilenetv3_model, compile_model, unfreeze_layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class AlzheimerDataset:
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['non_demented', 'very_mild_demented', 'mild_demented', 'moderate_demented']
        self.class_indices = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def load_data(self):
        images = []
        labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(self.class_indices[class_name])
        
        return np.array(images), np.array(labels)
    
    def create_data_generator(self, validation_split=0.2):
        """
        Create data generators with augmentation
        """
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        return datagen

def train_model(data_dir, model_save_path, epochs=50, batch_size=32):
    """
    Main training function
    """
    # Load dataset
    dataset = AlzheimerDataset(data_dir)
    images, labels = dataset.load_data()
    
    if len(images) == 0:
        raise ValueError("No images found in the dataset directory")
    
    # Convert labels to categorical
    labels = keras.utils.to_categorical(labels, num_classes=len(dataset.classes))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=np.argmax(labels, axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Create model
    model = create_mobilenetv3_model(num_classes=len(dataset.classes))
    model = compile_model(model)
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(model_save_path, save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Stage 1: Train with frozen base
    print("Stage 1: Training with frozen base layers...")
    history1 = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Stage 2: Fine-tuning with unfrozen layers
    print("Stage 2: Fine-tuning with unfrozen layers...")
    model = unfreeze_layers(model)
    
    history2 = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs // 2,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Generate predictions and classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=dataset.classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return model, history1, history2

if __name__ == "__main__":
    # Example usage
    data_directory = "path/to/your/dataset"  # Update this path
    model_path = "mobilenetv3_alzheimer.h5"
    
    try:
        model, history1, history2 = train_model(data_directory, model_path)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {str(e)}")