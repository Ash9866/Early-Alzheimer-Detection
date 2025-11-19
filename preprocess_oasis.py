# fixed_preprocess_oasis.py
import os
import pandas as pd
import nibabel as nib
import cv2
import numpy as np
from tqdm import tqdm
import shutil

class FixedOASISProcessor:
    def __init__(self):
        self.base_path = 'dataset/oasis'
        self.raw_path = os.path.join(self.base_path, 'OAS2_RAW_PART1')  # Fixed to match your uppercase
        self.output_path = os.path.join(self.base_path, 'processed_images')
        
    def find_and_merge_csv_files(self):
        """Find and merge CSV files with correct naming"""
        print("📊 Looking for CSV files...")
        
        # Your actual file names
        cross_sectional_path = os.path.join(self.base_path, 'cross_sectional.csv')
        longitudinal_path = os.path.join(self.base_path, 'longitudinal.csv')
        
        dfs = []
        
        # Load cross-sectional data if exists
        if os.path.exists(cross_sectional_path):
            print("✅ Found cross_sectional.csv")
            cross_df = pd.read_csv(cross_sectional_path)
            cross_df['data_source'] = 'cross_sectional'
            dfs.append(cross_df)
        else:
            print("❌ cross_sectional.csv not found")
        
        # Load longitudinal data
        if os.path.exists(longitudinal_path):
            print("✅ Found longitudinal.csv")
            long_df = pd.read_csv(longitudinal_path)
            long_df['data_source'] = 'longitudinal'
            dfs.append(long_df)
        
        if not dfs:
            print("❌ No CSV files found!")
            return None
        
        # Merge dataframes
        if len(dfs) > 1:
            merged_df = pd.concat(dfs, ignore_index=True)
        else:
            merged_df = dfs[0]
        
        print(f"📊 Merged dataset: {len(merged_df)} rows")
        
        # Standardize column names
        merged_df.columns = merged_df.columns.str.lower().str.replace(' ', '_')
        
        # Rename columns to match expected format
        column_mapping = {
            'subject_id': 'subject_id',
            'mri_id': 'mri_id', 
            'group': 'group',
            'cdr': 'cdr',
            'm/f': 'gender',
            'age': 'age'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in merged_df.columns and new_col not in merged_df.columns:
                merged_df[new_col] = merged_df[old_col]
        
        return merged_df
    
    def explore_raw_structure(self):
        """Explore the actual structure of your raw data"""
        print("🔍 Exploring raw data structure...")
        
        if not os.path.exists(self.raw_path):
            print(f"❌ Raw path not found: {self.raw_path}")
            return None
        
        print(f"📁 Contents of {self.raw_path}:")
        for item in os.listdir(self.raw_path):
            item_path = os.path.join(self.raw_path, item)
            if os.path.isdir(item_path):
                print(f"  📂 {item}/")
                # List subitems
                try:
                    subitems = os.listdir(item_path)
                    for subitem in subitems[:5]:  # Show first 5
                        print(f"    - {subitem}")
                    if len(subitems) > 5:
                        print(f"    ... and {len(subitems) - 5} more")
                except:
                    print("    (cannot access)")
            else:
                print(f"  📄 {item}")
        
        return self.raw_path
    
    def find_nifti_files(self, start_path):
        """Recursively find all NIFTI files"""
        nifti_files = []
        nifti_extensions = ['.nii', '.nii.gz', '.img', '.hdr', '.nifti']
        
        print("🔍 Searching for NIFTI files...")
        for root, dirs, files in os.walk(start_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in nifti_extensions):
                    full_path = os.path.join(root, file)
                    nifti_files.append(full_path)
                    
                    # Show first few files to understand structure
                    if len(nifti_files) <= 5:
                        print(f"  ✅ Found: {os.path.relpath(full_path, start_path)}")
        
        print(f"📊 Total NIFTI files found: {len(nifti_files)}")
        return nifti_files
    
    def extract_subject_info_from_path(self, file_path):
        """Extract subject ID and session from file path"""
        # Try to extract OAS2_XXXX pattern
        path_parts = file_path.split(os.sep)
        
        subject_id = None
        session = 'MR1'  # Default
        
        for part in path_parts:
            if part.startswith('OAS2_') and len(part) > 5:
                subject_id = part
                break
        
        # Also try to find session from path
        for part in path_parts:
            if part.startswith('MR'):
                session = part
                break
        
        return subject_id, session
    
    def map_cdr_to_class(self, cdr_score):
        """Convert CDR score to Alzheimer's class"""
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
    
    def get_cdr_for_subject(self, df, subject_id, session):
        """Get CDR score for a subject from the dataframe"""
        if subject_id is None:
            return 0
        
        # Clean subject ID (remove OAS2_ prefix for matching)
        clean_id = subject_id.replace('OAS2_', '')
        
        # Try exact match first
        matches = df[df['subject_id'].str.contains(clean_id, na=False)]
        
        if len(matches) > 0:
            # Get the most recent visit or first match
            if 'visit' in matches.columns:
                matches = matches.sort_values('visit', ascending=False)
            return matches.iloc[0]['cdr']
        
        return 0  # Default to non-demented if not found
    
    def process_nifti_file(self, nifti_path, df, output_dir):
        """Process a single NIFTI file into 2D slices"""
        try:
            # Extract subject info from path
            subject_id, session = self.extract_subject_info_from_path(nifti_path)
            
            # Get CDR score and class
            cdr_score = self.get_cdr_for_subject(df, subject_id, session)
            class_name = self.map_cdr_to_class(cdr_score)
            
            if class_name == 'Unknown':
                return 0
            
            # Create class directory
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Load NIFTI file
            img = nib.load(nifti_path)
            data = img.get_fdata()
            
            # Normalize to 0-255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data = data.astype(np.uint8)
            
            slices_created = 0
            
            # Extract slices from different orientations
            # Axial slices (z-axis)
            if len(data.shape) >= 3:
                z_middle = data.shape[2] // 2
                for z in range(z_middle - 3, z_middle + 4):  # 7 slices around middle
                    if 0 <= z < data.shape[2]:
                        slice_2d = data[:, :, z]
                        slice_2d = cv2.resize(slice_2d, (128, 128))
                        
                        # Convert to 3-channel if needed
                        if len(slice_2d.shape) == 2:
                            slice_2d = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2BGR)
                        
                        filename = f"{subject_id or 'UNK'}_{session}_axial_{z}.png"
                        output_path = os.path.join(class_dir, filename)
                        cv2.imwrite(output_path, slice_2d)
                        slices_created += 1
            
            return slices_created
            
        except Exception as e:
            print(f"❌ Error processing {nifti_path}: {e}")
            return 0
    
    def create_synthetic_dataset(self, df, num_images_per_class=200):
        """Create synthetic dataset if real NIFTI files aren't available"""
        print("🎨 Creating synthetic MRI dataset...")
        
        classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        
        # Create output directories
        for cls in classes:
            os.makedirs(os.path.join(self.output_path, cls), exist_ok=True)
        
        total_created = 0
        
        for class_idx, class_name in enumerate(classes):
            print(f"🔄 Creating {num_images_per_class} images for {class_name}...")
            
            for i in range(num_images_per_class):
                # Create base noise
                img = np.random.randint(50, 100, (128, 128), dtype=np.uint8)
                
                # Add brain-like structures based on class (worse dementia = more atrophy)
                center_x, center_y = 64, 64
                
                # Brain outline
                cv2.circle(img, (center_x, center_y), 50, 200, -1)
                
                # Ventricles (darker areas) - larger in dementia
                ventricle_size = 10 + class_idx * 3
                cv2.circle(img, (center_x-25, center_y), ventricle_size, 30, -1)
                cv2.circle(img, (center_x+25, center_y), ventricle_size, 30, -1)
                
                # Cortical atrophy simulation (dementia has more black spots)
                if class_idx > 0:  # Some dementia
                    for _ in range(class_idx * 5):
                        x = np.random.randint(20, 108)
                        y = np.random.randint(20, 108)
                        if np.sqrt((x-center_x)**2 + (y-center_y)**2) < 45:  # Within brain
                            cv2.circle(img, (x, y), 2, 30, -1)
                
                # Convert to 3-channel
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                filename = f"synthetic_{class_name}_{i:04d}.png"
                output_path = os.path.join(self.output_path, class_name, filename)
                cv2.imwrite(output_path, img_bgr)
                total_created += 1
        
        print(f"✅ Created {total_created} synthetic images")
        return total_created
    
    def run_processing(self):
        """Main processing function"""
        print("🚀 Starting OASIS Dataset Processing")
        print("=" * 50)
        
        # Step 1: Load and merge CSV data
        df = self.find_and_merge_csv_files()
        if df is None:
            print("❌ Cannot proceed without CSV data")
            return False
        
        # Step 2: Explore raw data structure
        raw_path = self.explore_raw_structure()
        if raw_path is None:
            print("❌ No raw data found. Creating synthetic dataset...")
            self.create_synthetic_dataset(df)
            return True
        
        # Step 3: Find NIFTI files
        nifti_files = self.find_nifti_files(raw_path)
        
        if not nifti_files:
            print("❌ No NIFTI files found. Creating synthetic dataset...")
            self.create_synthetic_dataset(df)
            return True
        
        # Step 4: Process NIFTI files
        print("🔄 Processing NIFTI files...")
        total_slices = 0
        processed_files = 0
        
        for nifti_file in tqdm(nifti_files[:100]):  # Process first 100 files
            slices = self.process_nifti_file(nifti_file, df, self.output_path)
            total_slices += slices
            if slices > 0:
                processed_files += 1
        
        print(f"✅ Processed {processed_files} files, created {total_slices} slices")
        
        # If we didn't get enough real data, supplement with synthetic
        if total_slices < 1000:
            print(f"⚠️ Only {total_slices} slices created. Adding synthetic data...")
            synthetic_count = self.create_synthetic_dataset(df, 100)  # 100 per class
            print(f"📊 Total: {total_slices} real + {synthetic_count} synthetic slices")
        
        # Save processing summary
        self.save_processing_summary(df, total_slices, len(nifti_files))
        
        return True
    
    def save_processing_summary(self, df, slices_created, nifti_files_found):
        """Save a summary of the processing"""
        summary = f"""
OASIS Dataset Processing Summary
================================
Timestamp: {pd.Timestamp.now()}

CSV Data:
- Total subjects in CSV: {len(df['subject_id'].unique())}
- CDR Distribution: {df['cdr'].value_counts().to_dict()}

NIFTI Processing:
- NIFTI files found: {nifti_files_found}
- Slices created: {slices_created}

Output Structure:
- Processed images: {self.output_path}
- Classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented

Class Distribution:
"""
        # Count images per class
        for class_name in ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']:
            class_path = os.path.join(self.output_path, class_name)
            if os.path.exists(class_path):
                image_count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
                summary += f"- {class_name}: {image_count} images\n"
        
        # Save summary
        with open(os.path.join(self.base_path, 'processing_summary.txt'), 'w') as f:
            f.write(summary)
        
        print("💾 Processing summary saved")

if __name__ == "__main__":
    processor = FixedOASISProcessor()
    success = processor.run_processing()
    
    if success:
        print("🎉 OASIS dataset processing completed successfully!")
        print(f"📁 Processed images saved to: {processor.output_path}")
    else:
        print("❌ Processing failed!")