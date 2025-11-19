# verify_dataset.py
import os

def verify_real_dataset():
    dataset_path = 'dataset/oasis/processed_images'
    
    print("🔍 Verifying dataset contents...")
    
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith('.png')]
            real_images = [f for f in images if 'synthetic' not in f]
            synthetic_images = [f for f in images if 'synthetic' in f]
            
            print(f"\n{cls}:")
            print(f"  Total images: {len(images)}")
            print(f"  Real MRI slices: {len(real_images)}")
            print(f"  Synthetic images: {len(synthetic_images)}")
            
            if real_images:
                print(f"  Sample real files: {real_images[:2]}")
            if synthetic_images:
                print(f"  ⚠️ Synthetic files found: {synthetic_images[:2]}")

if __name__ == "__main__":
    verify_real_dataset()