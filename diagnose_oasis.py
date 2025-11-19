# diagnose_oasis.py
import os
import pandas as pd
import tarfile

def diagnose_oasis_setup():
    print("🔍 Comprehensive OASIS Dataset Diagnosis")
    print("=" * 60)
    
    # Check basic folder structure
    base_path = 'dataset/oasis'
    print(f"📁 Checking base path: {base_path}")
    
    if not os.path.exists(base_path):
        print("❌ dataset/oasis folder doesn't exist!")
        print("💡 Create it with: mkdir -p dataset/oasis")
        return False
    
    # List all files in oasis folder
    print("\n📄 Files in dataset/oasis/:")
    files = os.listdir(base_path)
    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # Size in MB
            print(f"   {file} ({size:.1f} MB)")
        else:
            print(f"   📂 {file}/")
    
    # Check for required CSV files
    print("\n📊 Checking CSV files:")
    csv_files = {
        'cross-sectional': 'cross-sectional.csv',
        'longitudinal': 'longitudinal.csv'
    }
    
    missing_csv = []
    for name, filename in csv_files.items():
        csv_path = os.path.join(base_path, filename)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                print(f"✅ {name}: {len(df)} rows, {len(df.columns)} columns")
                print(f"   Columns: {list(df.columns)}")
                
                # Show first few rows
                print(f"   First 3 rows:")
                print(df.head(3).to_string())
                
            except Exception as e:
                print(f"❌ {name}: Error reading - {e}")
        else:
            print(f"❌ {name}: File not found")
            missing_csv.append(filename)
    
    # Check for OASIS data file
    print("\n📦 Checking OASIS raw data:")
    data_files = [
        'OAS2_RAW_part1.tar',
        'OAS2_RAW_part1.zip',
        'OAS2_RAW_part1'
    ]
    
    data_found = False
    for data_file in data_files:
        data_path = os.path.join(base_path, data_file)
        if os.path.exists(data_path):
            if os.path.isfile(data_path):
                size = os.path.getsize(data_path) / (1024*1024*1024)  # GB
                print(f"✅ Found: {data_file} ({size:.1f} GB)")
            else:
                print(f"✅ Found extracted folder: {data_file}")
            data_found = True
            break
    
    if not data_found:
        print("❌ No OASIS data file found!")
        print("💡 You need to download OAS2_RAW_part1.tar from OASIS website")
    
    # Check extracted structure
    print("\n📂 Checking extracted structure:")
    extracted_path = os.path.join(base_path, 'OAS2_RAW_part1')
    if os.path.exists(extracted_path):
        print(f"✅ Extracted folder exists")
        
        # Check for OAS2_RAW subfolder
        oasis_raw_path = os.path.join(extracted_path, 'OAS2_RAW')
        if os.path.exists(oasis_raw_path):
            subjects = [f for f in os.listdir(oasis_raw_path) if f.startswith('OAS2_')]
            print(f"✅ Found {len(subjects)} subject folders in OAS2_RAW")
            
            if subjects:
                # Check first subject
                sample_subject = subjects[0]
                sample_path = os.path.join(oasis_raw_path, sample_subject)
                print(f"📁 Sample subject: {sample_subject}")
                
                # List contents of sample subject
                if os.path.exists(sample_path):
                    sessions = [f for f in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, f))]
                    print(f"   Sessions: {sessions}")
                    
                    for session in sessions:
                        session_path = os.path.join(sample_path, session)
                        print(f"   📂 {session}:")
                        
                        # Check for RAW folder
                        raw_path = os.path.join(session_path, 'RAW')
                        if os.path.exists(raw_path):
                            files = os.listdir(raw_path)
                            nifti_files = [f for f in files if 'mpr' in f.lower() or '.nii' in f or '.img' in f]
                            print(f"     ✅ RAW folder: {len(nifti_files)} NIFTI files")
                            for nifti in nifti_files[:3]:  # Show first 3
                                print(f"        - {nifti}")
                        else:
                            print(f"     ❌ No RAW folder in {session}")
                            
                        # List all files in session
                        all_files = os.listdir(session_path)
                        print(f"     All files: {all_files}")
        else:
            print("❌ OAS2_RAW folder not found in extracted data!")
    else:
        print("❌ No extracted OASIS data found")
    
    # Check processed images
    print("\n🖼️ Checking processed images:")
    processed_path = os.path.join(base_path, 'processed_images')
    if os.path.exists(processed_path):
        classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        total_images = 0
        
        for cls in classes:
            class_path = os.path.join(processed_path, cls)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                print(f"   {cls}: {len(images)} images")
                total_images += len(images)
                if images:
                    print(f"     Sample: {images[0]}")
            else:
                print(f"   ❌ {cls}: Folder not found")
        
        print(f"📊 Total processed images: {total_images}")
    else:
        print("❌ No processed images folder found")
    
    return True

if __name__ == "__main__":
    diagnose_oasis_setup()