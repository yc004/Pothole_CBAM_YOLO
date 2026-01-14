import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Set encoding for Windows console
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul')

# Set style for better looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')

# Define paths
csv_path = 'Pothole_CBAM_Project/exp_cbam/results.csv'
output_dir = 'Pothole_CBAM_Project/exp_cbam'
output_file = os.path.join(output_dir, 'mAP_curve.png')

def plot_map():
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        # Read CSV
        # The file might have extra spaces in headers, so we strip them
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Check if required columns exist
        # Note: YOLOv8 CSV column names usually have (B) for box
        map50_col = 'metrics/mAP50(B)'
        map50_95_col = 'metrics/mAP50-95(B)'
        
        if map50_col not in df.columns:
            # Try finding column that contains mAP50
            candidates = [c for c in df.columns if 'mAP50' in c and '95' not in c]
            if candidates:
                map50_col = candidates[0]
        
        if map50_95_col not in df.columns:
             # Try finding column that contains mAP50-95
            candidates = [c for c in df.columns if 'mAP50-95' in c]
            if candidates:
                map50_95_col = candidates[0]

        print(f"Using columns: {map50_col} and {map50_95_col}")
        
        epochs = df['epoch']
        map50 = df[map50_col]
        map50_95 = df[map50_95_col]

        plt.figure(figsize=(12, 8))
        
        plt.plot(epochs, map50, label='mAP@50', linewidth=2, marker='.', markersize=5)
        plt.plot(epochs, map50_95, label='mAP@50-95', linewidth=2, marker='.', markersize=5)
        
        plt.title('Model Mean Average Precision (mAP) over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mAP', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate max values
        max_map50 = map50.max()
        max_map50_epoch = epochs[map50.idxmax()]
        plt.annotate(f'Max mAP@50: {max_map50:.4f}', 
                     xy=(max_map50_epoch, max_map50),
                     xytext=(max_map50_epoch, max_map50 - 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Successfully generated mAP chart at: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    plot_map()
