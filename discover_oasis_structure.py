# discover_oasis_structure.py
import os
import pandas as pd

def discover_real_structure():
    """Discover what your actual OASIS dataset structure looks like"""
    print("🔍 Discovering Your Actual OASIS Structure")
    print("=" * 60)
    
    base_path = 'dataset/oasis'
    raw_path = os.path.join(base_path, 'OAS2_RAW_PART1')
    
    if not os.path.exists(raw_path):
        print(f"❌ Raw path not found: {raw_path}")
        return
    
    print(f"📁 Exploring: {raw_path}")
    
    # Recursively explore and find all files
    nifti_files = []
    all_files = []
    
    for root, dirs, files in os.walk(raw_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, raw_path)
            all_files.append(rel_path)
            
            # Check if it's a NIFTI file
            if any(ext in file.lower() for ext in ['.nii', '.img', '.hdr', '.nifti']):
                nifti_files.append(rel_path)
    
    print(f"\n📊 Found {len(all_files)} total files")
    print(f"📊 Found {len(nifti_files)} NIFTI files")
    
    # Show directory structure
    print("\n📂 Directory Structure:")
    for i, file_path in enumerate(all_files[:50]):  # Show first 50 files
        print(f"  {file_path}")
    
    if len(all_files) > 50:
        print(f"  ... and {len(all_files) - 50} more files")
    
    # Show NIFTI files specifically
    if nifti_files:
        print(f"\n🧠 NIFTI Files Found:")
        for nifti_file in nifti_files[:20]:  # Show first 20 NIFTI files
            print(f"  ✅ {nifti_file}")
        
        if len(nifti_files) > 20:
            print(f"  ... and {len(nifti_files) - 20} more NIFTI files")
    else:
        print(f"\n❌ No NIFTI files found! Let's check what you actually have...")
        
        # List all file extensions to understand what you have
        print(f"\n📄 File Extensions Found:")
        extensions = {}
        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        for ext, count in sorted(extensions.items()):
            print(f"  {ext}: {count} files")
    
    return nifti_files, all_files

def check_csv_data():
    """Check what subject data we have in CSV files"""
    print(f"\n📊 Checking CSV Data:")
    
    longitudinal_path = 'dataset/oasis/longitudinal.csv'
    if os.path.exists(longitudinal_path):
        df = pd.read_csv(longitudinal_path)
        print(f"✅ Longitudinal CSV: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Unique subjects: {df['Subject ID'].nunique()}")
        print(f"   CDR scores: {df['CDR'].value_counts().to_dict()}")
        
        # Show sample of subject IDs
        sample_subjects = df['Subject ID'].head(10).tolist()
        print(f"   Sample subjects: {sample_subjects}")

if __name__ == "__main__":
    nifti_files, all_files = discover_real_structure()
    check_csv_data()