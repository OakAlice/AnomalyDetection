# Script to compare the performance of the models for different conditions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from MainScript import BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES

def plot_comparison(combined_metrics, DATASET_NAME, TARGET_ACTIVITIES, BASE_PATH):
        # Define my main colour palette for main 6 behaviours
        base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44"]

        print("Unique model types:", combined_metrics['model_type'].unique())
        
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
        model_order = ['oneclass', 'binary', 'multi_Other_NOthreshold', 
                      'multi_Activity_threshold', 'multi_Activity_NOthreshold']
        
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
        plot_path = Path(f"{BASE_PATH}/Output/Testing/Plots/{DATASET_NAME}_condition_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def combine_metrics(combined_metrics_path, DATASET_NAME):
    if combined_metrics_path.exists():
        combined_metrics = pd.read_csv(combined_metrics_path)
    else:            
        metrics_files = Path(f"{BASE_PATH}/Output/Testing/Metrics/").glob(f"{DATASET_NAME}*.csv")
        
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
            if filename_parts[3].lower() == "multi":
                metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}_{filename_parts[5].replace('_metrics.csv', '')}"
            else:
                metrics['thresholding'] = filename_parts[4].replace('_metrics.csv', '')
            
            # name the first column
            metrics = metrics.rename(columns={metrics.columns[0]: 'behaviour'})
            
            all_metrics.append(metrics)
        
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_metrics.to_csv(combined_metrics_path, index=False)

    return(combined_metrics)

def combine_confusion_matrices(BASE_PATH, DATASET_NAME):
    confusion_matrices_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/")
    
    # Define the expected layout
    training_sets = ['all', 'some', 'target']
    model_types = ['OneClass', 'Binary', 'Multi_Activity_NOthreshold', 
                  'Multi_Activity_threshold', 'Multi_Other_NOthreshold']
    
    # Create a figure with appropriate size
    fig, axes = plt.subplots(len(training_sets), len(model_types), 
                            figsize=(20, 12))
    
    # Iterate through all files and organize them into the grid
    for i, training_set in enumerate(training_sets):
        for j, model_type in enumerate(model_types):
            # Find the matching file
            pattern = f"{DATASET_NAME}*_{training_set}_{model_type}_*.png"
            matching_files = list(confusion_matrices_path.glob(pattern))
            

            if matching_files:
                # Read and display the confusion matrix
                conf_matrix = plt.imread(matching_files[0])
                axes[i, j].imshow(conf_matrix)
                axes[i, j].axis('off')
            else:
                # If no matching file, create empty subplot
                axes[i, j].axis('off')
    
    # Add row and column labels
    for i, training_set in enumerate(training_sets):
        axes[i, 0].set_ylabel(training_set, fontsize=12, rotation=90)
    
    for j, model_type in enumerate(model_types):
        axes[0, j].set_title(model_type, fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = Path(f"{BASE_PATH}/Output/Testing/Plots/{DATASET_NAME}combined_confusion_matrices.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main(BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES):
    # load in the metrics for each condition
    combined_metrics_path = Path(f"{BASE_PATH}/Output/Testing/{DATASET_NAME}_all_metrics.csv")

    print("combining the metrics")
    combined_metrics = combine_metrics(combined_metrics_path, DATASET_NAME)
    
    print("beginning dot plots")
    plot_comparison(combined_metrics, DATASET_NAME, TARGET_ACTIVITIES, BASE_PATH)

    print("beginning confusion matrix plots")
    combine_confusion_matrices(BASE_PATH, DATASET_NAME)

    print("done")

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES)
