# check_model_status.py
import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_model_status():
    """Check if the trained model is available and working"""
    print("🔍 Alzheimer's Model Status Check")
    print("=" * 50)
    
    # Check TensorFlow availability
    try:
        print(f"✅ TensorFlow Version: {tf.__version__}")
        tensorflow_available = True
    except:
        print("❌ TensorFlow not available")
        tensorflow_available = False
        return False
    
    # Check model files
    model_files = [
        'oasis_alzheimer_model.keras',
        'oasis_best_model.keras',
        'oasis_alzheimer_model.h5'
    ]
    
    model_found = False
    for model_file in model_files:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024*1024)  # MB
            print(f"✅ Model found: {model_file} ({file_size:.1f} MB)")
            model_found = True
            
            # Try to load the model
            try:
                model = tf.keras.models.load_model(model_file)
                print(f"✅ Model loaded successfully!")
                print(f"📊 Input shape: {model.input_shape}")
                print(f"📊 Output shape: {model.output_shape}")
                print(f"📊 Number of layers: {len(model.layers)}")
                return True
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
        else:
            print(f"❌ Model not found: {model_file}")
    
    if not model_found:
        print("❌ No trained model files found!")
        print("💡 Train the model first: python train_oasis_model_fixed.py")
        return False
    
    return False

def check_prediction_system():
    """Check the prediction system"""
    print("\n🎯 Prediction System Check")
    print("=" * 50)
    
    # Import here to avoid circular imports
    try:
        from predict_with_debug import load_predictor
    except ImportError:
        print("❌ Could not import prediction module")
        return False
    
    model_path = 'oasis_best_model.keras'
    if not os.path.exists(model_path):
        model_path = 'oasis_alzheimer_model.keras'
    
    try:
        predictor = load_predictor(model_path)
        
        print(f"📁 Model path: {model_path}")
        print(f"🧠 Model loaded: {predictor.model_loaded}")
        
        # Check TensorFlow availability separately
        try:
            import tensorflow as tf
            tf_available = True
            print(f"🔧 TensorFlow available: {tf_available} (v{tf.__version__})")
        except:
            tf_available = False
            print("🔧 TensorFlow available: False")
        
        if predictor.model_loaded:
            print("✅ System will use REAL MODEL predictions")
        else:
            print("❌ System will use MOCK PREDICTIONS")
            if not tf_available:
                print("   - Reason: TensorFlow not installed")
            elif not os.path.exists(model_path):
                print(f"   - Reason: Model file not found: {model_path}")
            else:
                print("   - Reason: Model failed to load")
            
        return predictor.model_loaded
        
    except Exception as e:
        print(f"❌ Error checking prediction system: {e}")
        return False

def quick_status_check():
    """Quick status check without detailed loading"""
    print("\n⚡ Quick System Status")
    print("=" * 30)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: v{tf.__version__}")
    except:
        print("❌ TensorFlow: Not installed")
        return False
    
    # Check model files
    model_files = ['oasis_best_model.keras', 'oasis_alzheimer_model.keras']
    model_exists = any(os.path.exists(f) for f in model_files)
    
    if model_exists:
        print("✅ Trained model: Found")
        # Quick load test
        try:
            model_path = next(f for f in model_files if os.path.exists(f))
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model load: Successful")
            print("🎯 Prediction: REAL MODEL")
            return True
        except Exception as e:
            print(f"❌ Model load: Failed ({e})")
            print("🎯 Prediction: MOCK (model load failed)")
            return False
    else:
        print("❌ Trained model: Not found")
        print("🎯 Prediction: MOCK (no model file)")
        return False

if __name__ == "__main__":
    # Run quick check first
    quick_status_check()
    
    print("\n" + "=" * 50)
    print("📋 Detailed System Analysis")
    print("=" * 50)
    
    model_ready = check_model_status()
    prediction_ready = check_prediction_system()
    
    print("\n" + "=" * 50)
    if model_ready and prediction_ready:
        print("🎉 SYSTEM STATUS: READY FOR REAL PREDICTIONS")
    else:
        print("⚠️  SYSTEM STATUS: WILL USE MOCK PREDICTIONS")
        print("\n💡 Solutions:")
        if not model_ready:
            print("   - Train the model: python train_oasis_model_fixed.py")
        else:
            print("   - Check model file permissions and paths")