#!/usr/bin/env python3
"""
Script to convert training logs from .npy format to .csv format
"""

import os
import sys
import argparse
import glob

try:
    import numpy as np
    import pandas as pd
    PACKAGES_AVAILABLE = True
except ImportError:
    PACKAGES_AVAILABLE = False


def convert_npy_to_csv_simple(npy_file, csv_file):
    """Convert .npy file to .csv using basic Python (fallback)"""
    try:
        # Read the npy file
        data = np.load(npy_file)
        
        # Define column names based on the training script
        columns = [
            'l_classification', 'l_distance', 'l_distance_masked', 
            'l_attention_fairness', 'loss_train',
            'acc_val', 'auc_roc_val', 'f1_val',
            'sp_val', 'eo_val',
            'p0_val', 'p1_val', 'REF_val',
            'v0_val', 'v1_val', 'VEF_val',
            'attention_jsd', 'comprehensive_score'
        ]
        
        # Handle different array shapes
        if len(data.shape) == 1:
            # Single row
            data = data.reshape(1, -1)
        
        # Adjust columns to match actual data width
        actual_cols = min(len(columns), data.shape[1])
        columns = columns[:actual_cols]
        
        # Create CSV content
        csv_content = ','.join(columns) + '\n'
        
        for row in data:
            row_str = ','.join([f'{val:.6f}' if isinstance(val, (int, float)) else str(val) 
                               for val in row[:actual_cols]])
            csv_content += row_str + '\n'
        
        # Write to CSV file
        with open(csv_file, 'w') as f:
            f.write(csv_content)
            
        return True
        
    except Exception as e:
        print(f"Error converting {npy_file}: {e}")
        return False


def convert_npy_to_csv_pandas(npy_file, csv_file):
    """Convert .npy file to .csv using pandas (preferred method)"""
    try:
        # Read the npy file
        data = np.load(npy_file)
        
        # Define column names
        columns = [
            'l_classification', 'l_distance', 'l_distance_masked', 
            'l_attention_fairness', 'loss_train',
            'acc_val', 'auc_roc_val', 'f1_val',
            'sp_val', 'eo_val',
            'p0_val', 'p1_val', 'REF_val',
            'v0_val', 'v1_val', 'VEF_val',
            'attention_jsd', 'comprehensive_score'
        ]
        
        # Handle different array shapes
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Adjust columns to match actual data width
        actual_cols = min(len(columns), data.shape[1])
        columns = columns[:actual_cols]
        
        # Create DataFrame
        df = pd.DataFrame(data[:, :actual_cols], columns=columns)
        
        # Save to CSV
        df.to_csv(csv_file, index=False, float_format='%.6f')
        
        return True
        
    except Exception as e:
        print(f"Error converting {npy_file}: {e}")
        return False


def find_npy_files(directory):
    """Find all .npy files in directory and subdirectories"""
    npy_files = []
    
    if os.path.isfile(directory) and directory.endswith('.npy'):
        return [directory]
    
    if os.path.isdir(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.npy'):
                    npy_files.append(os.path.join(root, file))
    
    return npy_files


def main():
    parser = argparse.ArgumentParser(description='Convert training log .npy files to .csv format')
    parser.add_argument('input', help='Input .npy file or directory containing .npy files')
    parser.add_argument('--output', '-o', help='Output directory (default: same as input)')
    parser.add_argument('--suffix', default='', help='Suffix to add to output filenames (default: none)')
    
    args = parser.parse_args()
    
    # Find all .npy files
    npy_files = find_npy_files(args.input)
    
    if not npy_files:
        print(f"No .npy files found in {args.input}")
        return
    
    print(f"Found {len(npy_files)} .npy file(s)")
    
    # Determine output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    # Convert files
    converted = 0
    failed = 0
    
    for npy_file in npy_files:
        print(f"Converting: {npy_file}")
        
        # Determine output filename
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        if args.suffix:
            base_name += args.suffix
        csv_name = base_name + '.csv'
        
        if output_dir:
            csv_file = os.path.join(output_dir, csv_name)
        else:
            csv_file = os.path.join(os.path.dirname(npy_file), csv_name)
        
        # Try conversion
        success = False
        if PACKAGES_AVAILABLE:
            success = convert_npy_to_csv_pandas(npy_file, csv_file)
        else:
            success = convert_npy_to_csv_simple(npy_file, csv_file)
        
        if success:
            print(f"  → {csv_file}")
            converted += 1
        else:
            failed += 1
    
    print(f"\nConversion complete: {converted} successful, {failed} failed")
    
    if not PACKAGES_AVAILABLE:
        print("\nNote: numpy and pandas not available, used fallback method")
        print("For better results, install: pip install numpy pandas")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode if no arguments
        print("Training Log Converter (.npy to .csv)")
        print("=" * 40)
        
        # Default: convert all logs in train_logs directory
        log_dirs = ['./train_logs', './train_logs/attention_fairness']
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                print(f"\nConverting logs in: {log_dir}")
                npy_files = find_npy_files(log_dir)
                
                for npy_file in npy_files:
                    csv_file = npy_file.replace('.npy', '.csv')
                    
                    if PACKAGES_AVAILABLE:
                        success = convert_npy_to_csv_pandas(npy_file, csv_file)
                    else:
                        success = convert_npy_to_csv_simple(npy_file, csv_file)
                    
                    if success:
                        print(f"✓ {os.path.basename(npy_file)} → {os.path.basename(csv_file)}")
                    else:
                        print(f"✗ Failed: {os.path.basename(npy_file)}")
        
        print(f"\nDone! Packages available: {PACKAGES_AVAILABLE}")
        
    else:
        # Command line mode
        main()