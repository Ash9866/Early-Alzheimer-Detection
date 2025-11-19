# simple_model_check.py
import os

def simple_model_check():
    print("🧠 Simple Alzheimer Model Check")
    print("=" * 40)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: v{tf.__version__}")
        tf_available = True
    except:
        print("❌ TensorFlow: Not available")
        tf_available = False
    
    # Check model files
    model_files = [
        'oasis_best_model.keras',
        'oasis_alzheimer_model.keras', 
        'oasis_alzheimer_model.h5'
    ]
    
    model_found = None
    for model_file in model_files:
        if os.path.exists(model_file):
            model_found = model_file
            size_mb = os.path.getsize(model_file) / (1024*1024)
            print(f"✅ Model file: {model_file} ({size_mb:.1f} MB)")
            break
    
    if not model_found:
        print("❌ Model file: Not found")
        print("\n💡 Run: python train_oasis_model_fixed.py")
        return "MOCK - No model file"
    
    # Try to load model
    if tf_available:
        try:
            model = tf.keras.models.load_model(model_found)
            print(f"✅ Model load: Successful")
            print(f"📊 Layers: {len(model.layers)}")
            print(f"🎯 Prediction: REAL MODEL")
            return "REAL MODEL"
        except Exception as e:
            print(f"❌ Model load: Failed - {e}")
            print("🎯 Prediction: MOCK - Load failed")
            return "MOCK - Load failed"
    else:
        print("🎯 Prediction: MOCK - No TensorFlow")
        return "MOCK - No TensorFlow"

if __name__ == "__main__":
    result = simple_model_check()
    print(f"\n🔍 FINAL STATUS: {result}")