import os
from config import Config

config = Config()

print("=== Model Path Verification ===")
print(f"Config MODEL_PATH: {config.MODEL_PATH}")
print(f"File exists: {os.path.exists(config.MODEL_PATH)}")

# Check both possible names
possible_names = [
    "app/deep_learning/models/mobilenetv3_alzheimer.h5",
    "app/deep_learning/models/mobilenetv3_model.h5"
]

print("\n=== Checking All Possible Model Files ===")
for path in possible_names:
    exists = os.path.exists(path)
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"{status}: {path}")

# Test the actual path from config
if os.path.exists(config.MODEL_PATH):
    print(f"\n🎉 SUCCESS: Model file found at configured path!")
    file_size = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB")
else:
    print(f"\n❌ PROBLEM: Model file not found at configured path!")
    print("Please either:")
    print("1. Update MODEL_PATH in config.py to point to the correct file")
    print("2. OR rename your model file to match the path in config.py")