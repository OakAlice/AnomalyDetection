# Script to compare the performance of the models for different conditions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from MainScript import BASE_PATH, ML_METHOD, DATASET_NAME, TARGET_ACTIVITIES

def plot_comparison(combined_metrics, TARGET_ACTIVITIES, BASE_PATH, ML_METHOD):
        # Define my main colour palette for main 6 behaviours
        base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44"]
        
        # Filter for just the behaviours of interest
        TARGET_ACTIVITIES = TARGET_ACTIVITIES + ['weighted_avg'] + ['Other']  # Convert to list concatenation
        combined_metrics = combined_metrics[combined_metrics['behaviour'].isin(TARGET_ACTIVITIES)]
        behaviours = combined_metrics['behaviour'].unique()
        print(behaviours)
        
        # Generate random colours for the other behaviours
        if len(behaviours) > len(base_colors):
            additional_colors = [f"#{np.random.randint(0, 0xFFFFFF):06x}" 
                               for _ in range(len(behaviours) - len(base_colors))]
            color_palette = base_colors + additional_colors
        else:
            color_palette = base_colors[:len(behaviours)]
        
        # Create color dictionary
        color_dict = dict(zip(behaviours, color_palette))
        
        # Melt the dataframe to get metrics in long format
        metrics_to_plot = ['F1', 'AUC', 'Precision', 'Recall']
        melted_metrics = pd.melt(combined_metrics,
                                id_vars=['dataset', 'model_type', 'training_set', 'thresholding', 'behaviour'],
                                value_vars=metrics_to_plot,
                                var_name='metric',
                                value_name='value')

        # Define the order for model_type
        model_order = ['OneClass', 'Binary', 'Multi_Other_NOthreshold', 
                      'Multi_Activity_threshold', 'Multi_Activity_NOthreshold']
        
        # Create faceted plot
        g = sns.FacetGrid(melted_metrics, 
                         col='training_set',
                         row='metric',
                         height=3,
                         aspect=1.5)

        # Draw points for non-weighted_avg data
        g.map_dataframe(lambda data, **kwargs: sns.scatterplot(
            data=data[data['behaviour'] != 'weighted_avg'].assign(
                model_type=lambda x: pd.Categorical(x['model_type'], categories=model_order, ordered=True)
            ),
            x='model_type',
            y='value',
            hue='behaviour',
            palette=color_dict,
            s=100))

        # Draw stars for weighted_avg data
        g.map_dataframe(lambda data, **kwargs: sns.scatterplot(
            data=data[data['behaviour'] == 'weighted_avg'].assign(
                model_type=lambda x: pd.Categorical(x['model_type'], categories=model_order, ordered=True)
            ),
            x='model_type',
            y='value',
            hue='behaviour',
            palette=color_dict,
            s=200,
            marker='*'))

        # Rotate x-axis labels for better readability and add column titles
        for ax in g.axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='x', labelrotation=45)
        
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        
        # Add y-axis grid lines for better readability
        g.map(plt.grid, axis='y', linestyle='--', alpha=0.7)

        # Add acix and plot labels
        g.set_axis_labels('Model Type', 'Value')

        # Add a legend outside the plot
        g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left')

        # Adjust layout to prevent label overlap
        plt.tight_layout()

        # Save the plot
        plot_path = Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Plots/condition_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    # load in the metrics for each condition
    metrics_files = Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Metrics/").glob("*.csv")
    all_metrics = []
    for file in metrics_files:
        metrics = pd.read_csv(file)
        # Add filename as a column
        metrics['file'] = file.name
        
        # Parse filename components correctly
        filename_parts = file.name.split('_')
        metrics['dataset'] = filename_parts[0] 
        metrics['training_set'] = filename_parts[2]
        metrics['model_type'] = filename_parts[3]
        if filename_parts[3] == "Multi":
            metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}_{filename_parts[5].replace('_metrics.csv', '')}"
        else:
            metrics['thresholding'] = filename_parts[4].replace('_metrics.csv', '')
        
        # name the first column
        metrics = metrics.rename(columns={metrics.columns[0]: 'behaviour'})
        
        all_metrics.append(metrics)
    
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/{DATASET_NAME}_all_metrics.csv", index=False)

    print("beginning plots")
    plot_comparison(combined_metrics, TARGET_ACTIVITIES, BASE_PATH, ML_METHOD)
    print("done")
