# debug_dataset.py
import os
import pandas as pd

def debug_dataset_structure():
    """Debug the dataset structure to find where the issue is"""
    
    print("🔍 Debugging Dataset Structure...")
    print("=" * 50)
    
    # Check processed images
    processed_path = 'dataset/oasis/processed_images'
    if os.path.exists(processed_path):
        print(f"✅ Processed images folder exists: {processed_path}")
        classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        
        total_images = 0
        for cls in classes:
            class_path = os.path.join(processed_path, cls)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                print(f"  {cls}: {len(images)} images")
                total_images += len(images)
            else:
                print(f"  ❌ {cls}: Folder not found")
        
        print(f"📊 Total images: {total_images}")
        
    else:
        print(f"❌ Processed images folder not found: {processed_path}")
    
    # Check CSV files
    print("\n📊 Checking CSV files...")
    csv_files = {
        'cross_sectional': 'dataset/oasis/cross-sectional.csv',
        'longitudinal': 'dataset/oasis/longitudinal.csv', 
        'merged': 'dataset/oasis/oasis_merged.csv'
    }
    
    for name, path in csv_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ {name}: {len(df)} rows, columns: {list(df.columns)}")
            
            # Check for CDR scores
            if 'cdr' in df.columns:
                cdr_counts = df['cdr'].value_counts()
                print(f"   CDR distribution: {cdr_counts.to_dict()}")
        else:
            print(f"❌ {name}: File not found")
    
    # Check raw data
    print("\n📁 Checking raw OASIS data...")
    raw_path = 'dataset/oasis/OAS2_RAW_part1'
    if os.path.exists(raw_path):
        print(f"✅ Raw data folder exists: {raw_path}")
        
        # Check for subject folders
        oasis_raw_path = os.path.join(raw_path, 'OAS2_RAW')
        if os.path.exists(oasis_raw_path):
            subjects = [f for f in os.listdir(oasis_raw_path) if f.startswith('OAS2_')]
            print(f"  Found {len(subjects)} subject folders")
            
            if subjects:
                # Check first subject
                sample_subject = subjects[0]
                sample_path = os.path.join(oasis_raw_path, sample_subject)
                sessions = [f for f in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, f))]
                print(f"  Sample subject '{sample_subject}' has sessions: {sessions}")
                
                # Check for NIFTI files
                for session in sessions:
                    session_path = os.path.join(sample_path, session)
                    raw_path_in_session = os.path.join(session_path, 'RAW')
                    if os.path.exists(raw_path_in_session):
                        nifti_files = [f for f in os.listdir(raw_path_in_session) if 'mpr' in f.lower()]
                        print(f"    Session {session}: {len(nifti_files)} NIFTI files")
    else:
        print(f"❌ Raw data folder not found: {raw_path}")

if __name__ == "__main__":
    debug_dataset_structure()