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

def get_map_columns(df):
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
            
    return map50_col, map50_95_col

def plot_single_model(csv_path, output_dir, model_name):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        output_file = os.path.join(output_dir, f'{model_name}_mAP_curve.png')
        
        # Read CSV
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        map50_col, map50_95_col = get_map_columns(df)
        print(f"[{model_name}] Using columns: {map50_col} and {map50_95_col}")
        
        epochs = df['epoch']
        map50 = df[map50_col]
        map50_95 = df[map50_95_col]

        plt.figure(figsize=(12, 8))
        
        plt.plot(epochs, map50, label='mAP@50', linewidth=2, marker='.', markersize=5)
        plt.plot(epochs, map50_95, label='mAP@50-95', linewidth=2, marker='.', markersize=5)
        
        plt.title(f'{model_name} - Mean Average Precision (mAP) over Epochs', fontsize=14)
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
        print(f"Successfully generated mAP chart for {model_name} at: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"An error occurred processing {model_name}: {e}")
        import traceback
        traceback.print_exc()

def plot_comparison(baseline_csv, improved_csv, output_dir):
    if not os.path.exists(baseline_csv) or not os.path.exists(improved_csv):
        print("Skipping comparison: One or both CSV files not found.")
        return

    try:
        output_file = os.path.join(output_dir, 'Comparison_mAP_curve.png')
        
        df_base = pd.read_csv(baseline_csv)
        df_base.columns = df_base.columns.str.strip()
        
        df_imp = pd.read_csv(improved_csv)
        df_imp.columns = df_imp.columns.str.strip()
        
        map50_col_base, _ = get_map_columns(df_base)
        map50_col_imp, _ = get_map_columns(df_imp)
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(df_base['epoch'], df_base[map50_col_base], label='Baseline mAP@50', linewidth=2, linestyle='--')
        plt.plot(df_imp['epoch'], df_imp[map50_col_imp], label='Improved (CBAM) mAP@50', linewidth=2)
        
        plt.title('Baseline vs Improved Model - mAP@50 Comparison', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mAP@50', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Annotate max values for both
        max_base = df_base[map50_col_base].max()
        max_imp = df_imp[map50_col_imp].max()
        
        plt.annotate(f'Baseline Max: {max_base:.4f}', 
                     xy=(df_base['epoch'][df_base[map50_col_base].idxmax()], max_base),
                     xytext=(10, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                     
        plt.annotate(f'Improved Max: {max_imp:.4f}', 
                     xy=(df_imp['epoch'][df_imp[map50_col_imp].idxmax()], max_imp),
                     xytext=(10, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Successfully generated Comparison chart at: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"An error occurred during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Configuration
    baseline_path = 'Pothole_Baseline_Project/exp_baseline/results.csv'
    cbam_path = 'Pothole_CBAM_Project/exp_cbam/results.csv'
    
    # Generate individual plots
    # Save baseline plot to its own folder
    plot_single_model(baseline_path, 'Pothole_Baseline_Project/exp_baseline', 'Baseline')
    
    # Save CBAM plot to its own folder
    plot_single_model(cbam_path, 'Pothole_CBAM_Project/exp_cbam', 'CBAM_Improved')
    
    # Generate comparison plot (saving in root or one of the folders, let's save in CBAM folder as it's the "result")
    plot_comparison(baseline_path, cbam_path, 'Pothole_CBAM_Project/exp_cbam')
