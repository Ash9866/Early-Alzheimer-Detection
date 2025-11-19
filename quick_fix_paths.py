# quick_fix_paths.py
import os
import shutil

def fix_paths():
    """Fix the path issues in your dataset"""
    
    # Fix 1: Rename cross_sectional.csv to match expected name
    if os.path.exists('dataset/oasis/cross_sectional.csv'):
        if not os.path.exists('dataset/oasis/cross-sectional.csv'):
            shutil.copy2('dataset/oasis/cross_sectional.csv', 'dataset/oasis/cross-sectional.csv')
            print("✅ Created cross-sectional.csv alias")
    
    # Fix 2: Check if we need to rename the raw folder
    uppercase_path = 'dataset/oasis/OAS2_RAW_PART1'
    lowercase_path = 'dataset/oasis/OAS2_RAW_part1'
    
    if os.path.exists(uppercase_path) and not os.path.exists(lowercase_path):
        os.rename(uppercase_path, lowercase_path)
        print("✅ Renamed OAS2_RAW_PART1 to OAS2_RAW_part1")
    
    print("🔧 Path fixes applied")

if __name__ == "__main__":
    fix_paths()