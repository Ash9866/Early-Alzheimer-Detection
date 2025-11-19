# real_oasis_processor.py
import os
import pandas as pd
import nibabel as nib
import cv2
import numpy as np
from tqdm import tqdm
import re

class RealOASISProcessor:
    def __init__(self):
        self.base_path = 'dataset/oasis'
        self.raw_path = os.path.join(self.base_path, 'OAS2_RAW_PART1')
        self.output_path = os.path.join(self.base_path, 'processed_images')
        
    def find_all_nifti_files(self):
        """Find ALL NIFTI files in your dataset"""
        print("🔍 Searching for NIFTI files in your dataset...")
        
        nifti_files = []
        nifti_extensions = ['.nii', '.nii.gz', '.img', '.hdr']
        
        for root, dirs, files in os.walk(self.raw_path):
            for file in files:
                file_lower = file.lower()
                if any(file_lower.endswith(ext) for ext in nifti_extensions):
                    full_path = os.path.join(root, file)
                    nifti_files.append(full_path)
        
        print(f"📊 Found {len(nifti_files)} NIFTI files")
        
        # Show some examples
        if nifti_files:
            print("📄 Sample NIFTI files:")
            for i, path in enumerate(nifti_files[:10]):
                rel_path = os.path.relpath(path, self.raw_path)
                print(f"  {i+1}. {rel_path}")
        
        return nifti_files
    
    def load_subject_data(self):
        """Load subject data from CSV files"""
        print("📊 Loading subject data from CSV...")
        
        longitudinal_path = os.path.join(self.base_path, 'longitudinal.csv')
        cross_sectional_path = os.path.join(self.base_path, 'cross_sectional.csv')
        
        dfs = []
        
        # Load longitudinal data
        if os.path.exists(longitudinal_path):
            long_df = pd.read_csv(longitudinal_path)
            long_df['data_source'] = 'longitudinal'
            dfs.append(long_df)
            print(f"✅ Loaded longitudinal data: {len(long_df)} rows")
        
        # Load cross-sectional data if available
        if os.path.exists(cross_sectional_path):
            cross_df = pd.read_csv(cross_sectional_path)
            cross_df['data_source'] = 'cross_sectional'
            dfs.append(cross_df)
            print(f"✅ Loaded cross-sectional data: {len(cross_df)} rows")
        
        if not dfs:
            print("❌ No CSV data found!")
            return None
        
        # Merge dataframes
        if len(dfs) > 1:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = dfs[0]
        
        # Clean column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        print(f"📊 Total merged data: {len(df)} rows")
        print(f"📈 CDR distribution: {df['cdr'].value_counts().to_dict()}")
        
        return df
    
    def extract_subject_session_from_path(self, nifti_path):
        """Extract subject ID and session from file path"""
        # Convert path to string and make it consistent
        path_str = str(nifti_path)
        
        # Look for OAS2_XXXX pattern
        subject_match = re.search(r'OAS2_(\d+)', path_str, re.IGNORECASE)
        if subject_match:
            subject_id = f"OAS2_{subject_match.group(1)}"
        else:
            # Try alternative patterns
            alt_match = re.search(r'(\d+)_MR', path_str)
            if alt_match:
                subject_id = f"OAS2_{alt_match.group(1).zfill(4)}"
            else:
                print(f"⚠️ Could not extract subject ID from: {path_str}")
                return None, None
        
        # Look for session (MR1, MR2, etc.)
        session_match = re.search(r'(MR\d+)', path_str, re.IGNORECASE)
        if session_match:
            session = session_match.group(1).upper()
        else:
            session = 'MR1'  # Default
        
        return subject_id, session
    
    def get_cdr_for_subject(self, df, subject_id, session):
        """Get CDR score for a subject"""
        if subject_id is None:
            return 0.0
        
        # Clean subject ID for matching
        clean_id = subject_id.replace('OAS2_', '').lstrip('0')
        
        # Try to find matching subject
        matches = df[
            (df['subject_id'].str.contains(clean_id, na=False)) | 
            (df['mri_id'].str.contains(clean_id, na=False)) |
            (df['subject_id'].str.contains(subject_id, na=False))
        ]
        
        if len(matches) > 0:
            # If multiple matches, try to match session
            if 'mri_id' in matches.columns:
                session_matches = matches[matches['mri_id'].str.contains(session, na=False)]
                if len(session_matches) > 0:
                    return float(session_matches.iloc[0]['cdr'])
            
            # Return first match's CDR
            return float(matches.iloc[0]['cdr'])
        
        print(f"⚠️ No CDR data found for {subject_id} {session}")
        return 0.0
    
    def map_cdr_to_class(self, cdr_score):
        """Map CDR score to Alzheimer's class"""
        if pd.isna(cdr_score):
            return 'NonDemented'
        
        cdr = float(cdr_score)
        if cdr == 0:
            return 'NonDemented'
        elif cdr == 0.5:
            return 'VeryMildDemented'
        elif cdr == 1:
            return 'MildDemented'
        elif cdr >= 2:
            return 'ModerateDemented'
        else:
            return 'Unknown'
    
    def process_nifti_to_slices(self, nifti_path, subject_id, session, class_name):
        """Convert 3D NIFTI to 2D slices"""
        try:
            # Load NIFTI file
            img = nib.load(nifti_path)
            data = img.get_fdata()
            
            print(f"📐 NIFTI shape: {data.shape}")
            
            # Normalize to 0-255
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:  # Avoid division by zero
                data = (data - data_min) / (data_max - data_min) * 255
            data = data.astype(np.uint8)
            
            slices_created = 0
            
            # Create output directory
            class_dir = os.path.join(self.output_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Extract slices from different dimensions
            # Assuming the data is 3D: (x, y, z) or (x, y, z, time)
            if len(data.shape) >= 3:
                # Use the first three dimensions
                dim1, dim2, dim3 = data.shape[0], data.shape[1], data.shape[2]
                
                # Axial slices (z-axis)
                for z in [dim3//4, dim3//2, 3*dim3//4]:  # Quarter, middle, three-quarters
                    if 0 <= z < dim3:
                        slice_data = data[:, :, z]
                        slice_resized = cv2.resize(slice_data, (128, 128))
                        
                        # Ensure 3 channels
                        if len(slice_resized.shape) == 2:
                            slice_resized = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2BGR)
                        
                        filename = f"{subject_id}_{session}_axial_{z}.png"
                        output_path = os.path.join(class_dir, filename)
                        cv2.imwrite(output_path, slice_resized)
                        slices_created += 1
                
                # Sagittal slices (x-axis)
                for x in [dim1//4, dim1//2, 3*dim1//4]:
                    if 0 <= x < dim1:
                        slice_data = data[x, :, :]
                        slice_resized = cv2.resize(slice_data, (128, 128))
                        
                        if len(slice_resized.shape) == 2:
                            slice_resized = cv2.cvtColor(slice_resized, cv2.COLOR_GRAY2BGR)
                        
                        filename = f"{subject_id}_{session}_sagittal_{x}.png"
                        output_path = os.path.join(class_dir, filename)
                        cv2.imwrite(output_path, slice_resized)
                        slices_created += 1
            
            print(f"✅ Created {slices_created} slices for {subject_id}")
            return slices_created
            
        except Exception as e:
            print(f"❌ Error processing {nifti_path}: {e}")
            return 0
    
    def process_real_dataset(self):
        """Process the REAL OASIS dataset (no synthetic images)"""
        print("🚀 PROCESSING REAL OASIS DATASET")
        print("=" * 50)
        
        # Step 1: Load subject data
        df = self.load_subject_data()
        if df is None:
            print("❌ Cannot proceed without subject data")
            return False
        
        # Step 2: Find all NIFTI files
        nifti_files = self.find_all_nifti_files()
        if not nifti_files:
            print("❌ No NIFTI files found in your dataset!")
            print("💡 Please make sure your OAS2_RAW_PART1 folder contains NIFTI files")
            return False
        
        # Step 3: Process each NIFTI file
        print(f"\n🔄 Processing {len(nifti_files)} NIFTI files...")
        total_slices = 0
        processed_files = 0
        
        for nifti_path in tqdm(nifti_files):
            try:
                # Extract subject info from path
                subject_id, session = self.extract_subject_session_from_path(nifti_path)
                
                if subject_id is None:
                    continue
                
                # Get CDR score and class
                cdr_score = self.get_cdr_for_subject(df, subject_id, session)
                class_name = self.map_cdr_to_class(cdr_score)
                
                if class_name == 'Unknown':
                    continue
                
                # Process the NIFTI file
                slices_created = self.process_nifti_to_slices(
                    nifti_path, subject_id, session, class_name
                )
                
                if slices_created > 0:
                    total_slices += slices_created
                    processed_files += 1
                    
                    # Show progress every 10 files
                    if processed_files % 10 == 0:
                        print(f"📊 Progress: {processed_files}/{len(nifti_files)} files, {total_slices} slices")
                        
            except Exception as e:
                print(f"❌ Failed to process {nifti_path}: {e}")
                continue
        
        # Step 4: Report results
        print(f"\n🎉 PROCESSING COMPLETED!")
        print(f"📊 Files processed: {processed_files}/{len(nifti_files)}")
        print(f"📊 Total slices created: {total_slices}")
        print(f"📁 Output directory: {self.output_path}")
        
        # Show class distribution
        print(f"\n📈 Class Distribution:")
        classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        for class_name in classes:
            class_path = os.path.join(self.output_path, class_name)
            if os.path.exists(class_path):
                image_count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
                print(f"  {class_name}: {image_count} images")
        
        return True

if __name__ == "__main__":
    processor = RealOASISProcessor()
    success = processor.process_real_dataset()
    
    if success:
        print("\n✅ REAL OASIS dataset processing completed!")
        print("🎯 Now you can run: python train_oasis_model_fixed.py")
    else:
        print("\n❌ Processing failed!")
        print("💡 Please check that your OASIS dataset contains NIFTI files")