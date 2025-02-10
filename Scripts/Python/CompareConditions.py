# Script to compare the performance of the models for different conditions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from MainScript import BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES

def plot_comparison(combined_metrics, DATASET_NAME, TARGET_ACTIVITIES, BASE_PATH):
        base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44", 
                      "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6", 
                      "#FF8C94", "#86A8E7", "#D4A5A5", "#9ED9CC", "#FFE156", "#8B4513"]

        print("Unique model types:", combined_metrics['model_type'].unique())
        
        behaviours = combined_metrics['behaviour'].unique()
        print(behaviours)
        
        # Create colour dictionary
        colour_dict = dict(zip(behaviours, base_colors))
        
        # Melt the dataframe to get metrics in long format
        metrics_to_plot = ['F1', 'AUC', 'Precision', 'Recall']
        melted_metrics = pd.melt(combined_metrics,
                                id_vars=['dataset', 'model_type', 'training_set', 'behaviour'],
                                value_vars=metrics_to_plot,
                                var_name='metric',
                                value_name='value')

        # Define the order for model_type
        model_order = ['multi_Activity_NOthreshold', 'oneclass', 'binary', 'multi_Other', 
                      'multi_Activity_threshold' ]
        
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
            palette=colour_dict,
            s=100))

        # Draw stars for weighted_avg data
        g.map_dataframe(lambda data, **kwargs: sns.scatterplot(
            data=data[data['behaviour'] == 'weighted_avg'].assign(
                model_type=lambda x: pd.Categorical(x['model_type'], categories=model_order, ordered=True)
            ),
            x='model_type',
            y='value',
            hue='behaviour',
            palette=colour_dict,
            s=200,
            marker='*'))

        # Rotate x-axis labels for better readability and add column titles
        for ax in g.axes.flat:
            # Set y-axis limits from 0 to 1 (removing negative space)
            ax.set_ylim(0, 1.0)
            
            # Remove break lines and negative space shading
            # (removing the ax.axhline, ax.axhspan, and ax.plot calls)
            
            # Adjust y-ticks to show regular scale
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            
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

def plot_auc_comparison(base_path):
    """
    Creates a faceted plot comparing AUC scores across datasets and training conditions.
    Datasets are shown as rows and training conditions as columns.
    
    Args:
        base_path (str): Base path where the metrics files are stored
    """
    # Combine metrics from all files
    combined_metrics_path = Path(f"{base_path}/Output/Testing/")
    all_metrics = []
    for file in combined_metrics_path.glob("*.csv"):
        metrics = pd.read_csv(file)
        all_metrics.append(metrics)
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    save_path = Path(f"{base_path}/Output/Testing/all_combined_metrics.csv")
    combined_metrics.to_csv(save_path, index=False)
    

    # Prepare data for plotting
    metrics_to_plot = ['AUC']
    melted_metrics = pd.melt(combined_metrics,
                            id_vars=['dataset', 'model_type', 'training_set', 'behaviour'],
                            value_vars=metrics_to_plot,
                            var_name='metric',
                            value_name='value')
    
    # Define the order for model_type
    model_order = ['multi_Activity_NOthreshold', 'oneclass', 'binary', 'multi_Other', 
                   'multi_Activity_threshold']
    
    # Create faceted plot with datasets as rows and training_set as columns
    g = sns.FacetGrid(melted_metrics, 
                      row='dataset',
                      col='training_set',
                      height=4,
                      aspect=1.5)
    
    # Draw points for non-weighted_avg data using default rainbow colors
    g.map_dataframe(lambda data, **kwargs: sns.scatterplot(
        data=data[data['behaviour'] != 'weighted_avg'].assign(
            model_type=lambda x: pd.Categorical(x['model_type'], categories=model_order, ordered=True)
        ),
        x='model_type',
        y='value',
        hue='behaviour',
        palette='rainbow',  # Changed to rainbow palette
        s=100))
    
    # Draw stars for weighted_avg data
    g.map_dataframe(lambda data, **kwargs: sns.scatterplot(
        data=data[data['behaviour'] == 'weighted_avg'].assign(
            model_type=lambda x: pd.Categorical(x['model_type'], categories=model_order, ordered=True)
        ),
        x='model_type',
        y='value',
        hue='behaviour',
        palette=['black'],  # Changed all weighted aberages to black
        s=200,
        marker='*'))
    
    # Customize plot appearance
    for ax in g.axes.flat:
        # Set y-axis limits to show full range from 0 to 1
        ax.set_ylim(0, 1.0)
        
        # Add y-ticks at regular intervals
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelrotation=45)
    
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.map(plt.grid, axis='y', linestyle='--', alpha=0.7)
    g.set_axis_labels('Model Type', 'AUC Score')
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = Path(f"{base_path}/Output/Testing/Plots/dataset_comparison_AUC.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

def elapsed_time(base_path):
    # load in the combined_optimisation_results.csv
    combined_optimisation_results = pd.read_csv(Path(f"{base_path}/Output/Tuning/Combined_optimisation_results.csv"))

    # Define the same orders as in plot_auc_comparison for consistency
    model_order = ['OneClass', 'Binary', 'Multi_Activity_NOthreshold', 
                  'Multi_Activity_threshold', 'Multi_Other_NOthreshold']
    training_set_order = ['all', 'some', 'target']
    dataset_order = ['Ferdinandy_Dog', 'Vehkaoja_Dog']

    # Create figure with FacetGrid
    g = sns.FacetGrid(
        combined_optimisation_results,
        col='dataset_name',
        row='training_set',
        margin_titles=True,
        height=4,
        aspect=1.5
    )

    # Plot elapsed time points
    g.map_dataframe(
        sns.scatterplot,
        data=combined_optimisation_results.assign(
            model_type=lambda x: pd.Categorical(x['model_type'], categories=model_order, ordered=True)
        ),
        x='model_type',
        y='elapsed_time',
        hue='behaviour',
        s=100
    )

    # Customize plot appearance
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels('Model Type', 'Elapsed Time (seconds)')
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left')

    # Adjust layout and save
    plt.tight_layout()
    plot_path = Path(f"{base_path}/Output/Testing/Plots/dataset_comparison_time.png")
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
                if "Other" in filename_parts[4]:
                    metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}"
                else:
                    metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}_{filename_parts[5].replace('_metrics.csv', '')}"
        
            # name the first column
            metrics = metrics.rename(columns={metrics.columns[0]: 'behaviour'})
            
            all_metrics.append(metrics)
        
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_metrics.to_csv(combined_metrics_path, index=False)
        print(f"combined_metrics {combined_metrics}")

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
