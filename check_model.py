# Create a quick check script: check_model.py
import os

model_path = "app/deep_learning/models/mobilenetv3_alzheimer.h5"
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    print(f"Model file size: {file_size:.2f} MB")
else:
    print("❌ Model file not found!")
    print("Looking for model in other locations...")
    
    # Check common locations
    possible_locations = [
        "mobilenetv3_alzheimer.h5",
        "final_model.h5", 
        "improved_mobilenetv3_alzheimer.h5",
        "app/deep_learning/mobilenetv3_alzheimer.h5"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"✅ Found model at: {location}")
            print(f"Copy it to: {model_path}")
            break
    else:
        print("❌ No model file found anywhere!")