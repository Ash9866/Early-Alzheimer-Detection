# predict_with_debug.py
import os
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    logger.info(f"✅ TensorFlow {tf.__version__} imported successfully")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(f"❌ TensorFlow not available: {e}")

class DebugAlzheimerPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        self.img_size = (128, 128)
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load model with detailed debugging"""
        logger.info(f"🔧 Attempting to load model from: {self.model_path}")
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("❌ TensorFlow not available - will use mock predictions")
            self.model_loaded = False
            return
            
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Model file not found: {self.model_path}")
            self.model_loaded = False
            return
        
        try:
            logger.info("🔄 Loading model...")
            self.model = keras.models.load_model(self.model_path)
            self.model_loaded = True
            logger.info("✅ Model loaded successfully!")
            
            # Print model summary
            logger.info("📊 Model Summary:")
            self.model.summary(print_fn=lambda x: logger.info(f"  {x}"))
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.model_loaded = False
            self.model = None
    
    def predict(self, image_path):
        """Make prediction with detailed logging"""
        logger.info(f"🎯 Starting prediction for: {image_path}")
        
        # Check if model is available
        if not self.model_loaded or self.model is None:
            logger.warning("🚨 Using MOCK PREDICTION - Model not loaded")
            return self.mock_predict(image_path)
        
        try:
            # Preprocess image
            logger.info("🔄 Preprocessing image...")
            processed_img = self.preprocess_image(image_path)
            
            # Make prediction
            logger.info("🧠 Making prediction with real model...")
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Process results
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            probabilities = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            result = {
                'predicted_class': self.class_names[predicted_class_idx],
                'confidence': float(confidence),
                'probabilities': probabilities,
                'success': True,
                'method': 'REAL_MODEL',
                'model_used': os.path.basename(self.model_path)
            }
            
            logger.info(f"✅ REAL MODEL PREDICTION: {result['predicted_class']} "
                       f"(confidence: {result['confidence']:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Real model prediction failed: {e}")
            logger.warning("🔄 Falling back to mock prediction...")
            result = self.mock_predict(image_path)
            result['error'] = str(e)
            return result
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def mock_predict(self, image_path):
        """Mock prediction with clear identification"""
        import random
        
        logger.warning("🤖 Using MOCK PREDICTION SYSTEM")
        
        predicted_class = random.choice(self.class_names)
        confidence = random.uniform(0.6, 0.9)
        
        # Generate probabilities
        base_probs = [random.random() for _ in self.class_names]
        predicted_idx = self.class_names.index(predicted_class)
        base_probs[predicted_idx] += 1.0
        total = sum(base_probs)
        probabilities = {cls: prob/total for cls, prob in zip(self.class_names, base_probs)}
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'success': True,
            'method': 'MOCK_PREDICTION',
            'warning': 'This is a mock prediction - model not available',
            'model_used': 'None'
        }
        
        logger.info(f"🤖 MOCK PREDICTION: {result['predicted_class']} "
                   f"(confidence: {result['confidence']:.3f})")
        return result

# Global instance with debugging
_predictor = None

def load_predictor(model_path):
    global _predictor
    if _predictor is None:
        logger.info(f"🔄 Creating new predictor instance with model: {model_path}")
        _predictor = DebugAlzheimerPredictor(model_path)
    return _predictor

def predict_alzheimer_debug(image_path, model_path=None):
    """Main prediction interface with debugging"""
    if model_path is None:
        from flask import current_app
        model_path = current_app.config['MODEL_PATH']
    
    logger.info("=" * 50)
    logger.info(f"🚀 PREDICTION REQUEST: {image_path}")
    logger.info(f"📁 Model path: {model_path}")
    
    predictor = load_predictor(model_path)
    result = predictor.predict(image_path)
    
    logger.info(f"🎯 FINAL RESULT: {result['method']} - {result['predicted_class']}")
    logger.info("=" * 50)
    
    return result

# Test function
def test_prediction_system():
    """Test the prediction system with a sample image"""
    print("🧪 Testing Prediction System")
    print("=" * 50)
    
    model_path = 'oasis_alzheimer_model.keras'  # or 'oasis_best_model.keras'
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"✅ Model file exists: {model_path}")
        file_size = os.path.getsize(model_path) / (1024*1024)  # MB
        print(f"📦 Model size: {file_size:.1f} MB")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("💡 Train the model first: python train_oasis_model_fixed.py")
        return
    
    # Find a test image
    test_images = []
    for root, dirs, files in os.walk('dataset/oasis/processed_images'):
        for file in files:
            if file.endswith('.png'):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 3:
                    break
        if test_images:
            break
    
    if test_images:
        test_image = test_images[0]
        print(f"🧪 Using test image: {test_image}")
        
        # Make prediction
        result = predict_alzheimer_debug(test_image, model_path)
        
        print(f"\n📊 PREDICTION RESULT:")
        print(f"   Method: {result['method']}")
        print(f"   Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model Used: {result.get('model_used', 'Unknown')}")
        
        if 'warning' in result:
            print(f"   ⚠️  {result['warning']}")
            
    else:
        print("❌ No test images found in processed_images folder")

if __name__ == "__main__":
    test_prediction_system()