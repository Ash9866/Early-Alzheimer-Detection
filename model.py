import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV3Large
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

def create_mobilenetv3_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Create MobileNetV3 model for Alzheimer's classification
    """
    # Load pre-trained MobileNetV3Large
    base_model = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate settings
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

def unfreeze_layers(model, unfreeze_after=100):
    """
    Unfreeze layers for fine-tuning
    """
    # Unfreeze the top layers
    for layer in model.layers[-unfreeze_after:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model