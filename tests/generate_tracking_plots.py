import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_tracking_plot_from_csv(csv_file):
    """Generate tracking plots from existing CSV data."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if we have the required columns
        required_columns = ['t', 'theta', 'theta_target']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in {csv_file}")
            print(f"Available columns: {list(df.columns)}")
            return
            
        # Extract data
        time = df['t'].values
        theta_actual = df['theta'].values
        theta_target = df['theta_target'].values
        
        # Calculate tracking error
        tracking_error = np.abs(theta_actual - theta_target)
        avg_error = np.mean(tracking_error)
        max_error = np.max(tracking_error)
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate tracking performance plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time, theta_actual, label='Actual Theta')
        plt.plot(time, theta_target, label='Target Theta', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Theta (rad)')
        plt.title(f'Position Control Tracking Performance\n{csv_file.name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, tracking_error, 'r', label='Tracking Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (rad)')
        plt.title('Tracking Error Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        plt.text(0.02, 0.98, f'Avg Error: {avg_error:.4f}\nMax Error: {max_error:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = plots_dir / f"{csv_file.stem}_tracking_plot.png"
        plt.savefig(plot_file, dpi=120)
        plt.close()
        
        print(f"Generated tracking plot for {csv_file.name}")
        print(f"  Average error: {avg_error:.4f}")
        print(f"  Maximum error: {max_error:.4f}")
        print(f"  Saved to: {plot_file}")
        print()
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def main():
    """Generate tracking plots for all CSV files."""
    print("Generating tracking plots from existing CSV files...")
    
    # Find all CSV files in output directory
    output_dir = Path("output")
    csv_files = list(output_dir.glob("**/*.csv"))
    
    if not csv_files:
        print("No CSV files found in output directory")
        return
        
    print(f"Found {len(csv_files)} CSV files")
    print()
    
    # Generate plots for each CSV file
    for csv_file in csv_files:
        generate_tracking_plot_from_csv(csv_file)

if __name__ == "__main__":
    main()