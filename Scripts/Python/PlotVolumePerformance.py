import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def PlotVolumePerformance(BASE_PATH):
    # load in the data
    metrics_path = Path(f"{BASE_PATH}/Output/Testing/all_combined_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    
    # Filter out rows with 'weighted_avg' behaviour
    metrics = metrics[metrics['behaviour'] != 'weighted_avg']
    metrics = metrics[metrics['model_type'] == 'multi_Activity_NOthreshold']
    metrics = metrics[metrics['training_set'] == 'all']
    
    # Import numpy to generate the rainbox colour map
    import numpy as np
    
    # Get all unique behaviours and generate a colour for each from the 'rainbow' colormap
    unique_behaviours = metrics['behaviour'].unique()
    colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(unique_behaviours)))
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Loop over each unique behaviour and plot with its corresponding colour
    for i, behaviour in enumerate(unique_behaviours):
        subset = metrics[metrics['behaviour'] == behaviour]
        plt.scatter(subset['Support'], subset['F1'], label=behaviour, color=colors[i])
    
    plt.xlabel('Volume')
    plt.ylabel('F1')
    plt.title('Volume to Performance by Behaviour')
    plt.legend(title='Behaviour')
    
    # Save the plot as a PDF
    plt.savefig(f"{BASE_PATH}/Output/Testing/Plots/volume_performance_by_behaviour.pdf")


def main(BASE_PATH):
    PlotVolumePerformance(BASE_PATH)


