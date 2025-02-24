# Script to compare the performance of the models for different conditions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from MainScript import BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES

def plot_comparison(combined_metrics, TARGET_ACTIVITIES, BASE_PATH):
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
        plot_path = Path(f"{BASE_PATH}/Output/Testing/Plots/condition_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

def combine_metrics(combined_metrics_path):
    if combined_metrics_path.exists():
        combined_metrics = pd.read_csv(combined_metrics_path)
    else:
        metrics_files = Path(f"{BASE_PATH}/Output/Testing/Metrics/").glob("*.csv")
            
        metrics_files = Path(f"{BASE_PATH}/Output/Testing/Metrics/").glob("*.csv")
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
        combined_metrics.to_csv(combined_metrics_path, index=False)

    return(combined_metrics)

def combine_confusion_matrices(BASE_PATH):
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
            pattern = f"*_{training_set}_{model_type}_*.png"
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
    save_path = Path(f"{BASE_PATH}/Output/Testing/Plots/combined_confusion_matrices.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_false_predictions(BASE_PATH, DATASET_NAME):
    """
    This function plots the false predictions for each model type and training set in a grid layout
    Each subplot is a stacked bar chart with the predicted class along the x axis and the count of false positives on the y axis, 
    coloured by the true class
    """
    confusion_matrices_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices")
    print(confusion_matrices_path)
    pattern = f"{DATASET_NAME}_*.csv"
    matching_files = list(confusion_matrices_path.glob(pattern))
    
    # Check if any files were found
    if not matching_files:
        print(f"No confusion matrix CSV files found matching pattern: {pattern}")
        return
    
    # Calculate grid dimensions
    n_files = len(matching_files)
    n_cols = min(5, n_files)  # Maximum 5 columns, minimum 1
    n_rows = (n_files + n_cols - 1) // n_cols  # Ceiling division for number of rows
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])  # Convert single axis to array for consistent indexing
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Create plots
    for idx, (file, ax) in enumerate(zip(matching_files, axes)):
        df = pd.read_csv(file)
        df_long = df.melt(id_vars=df.columns[0], value_vars=df.columns[1:], 
                         var_name='predicted', value_name='count')
        df_long = df_long.rename(columns={df_long.columns[0]: 'true'})
        
        # Create bar chart
        sns.barplot(x='predicted', y='count', hue='true', data=df_long, ax=ax)
        
        # Customize subplot
        title = file.stem.replace('_confusion_matrix', '')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Count')
    
    # Remove empty subplots if any
    for idx in range(len(matching_files), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_path = Path(f"{BASE_PATH}/Output/Testing/Plots/{DATASET_NAME}_false_predictions_combined.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(BASE_PATH, DATASET_NAME):
    # load in the metrics for each condition
    # combined_metrics_path = Path(f"{BASE_PATH}/Output/Testing/{DATASET_NAME}_all_metrics.csv")

    # print("combining the metrics")
    # combined_metrics = combine_metrics(combined_metrics_path)
    
    # print("beginning dot plots")
    # plot_comparison(combined_metrics, TARGET_ACTIVITIES, BASE_PATH)

    # print("beginning confusion matrix plots")
    # combine_confusion_matrices(BASE_PATH, DATASET_NAME)

    print("beginning false predictions plots")
    plot_false_predictions(BASE_PATH, DATASET_NAME)

    print("done")


if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME)
