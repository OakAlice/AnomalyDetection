# Script to compare the performance of the models for different conditions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from MainScript import BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES

def plot_model_type_comparison(combined_metrics, DATASET_NAME, TARGET_ACTIVITIES, BASE_PATH):
    # remove error from the combined_metrics
    plot_behaviours = TARGET_ACTIVITIES + ['Other', 'weighted_avg']
    print(f"plot_behaviours: {plot_behaviours}")
    combined_metrics = combined_metrics[combined_metrics['behaviour'].isin(plot_behaviours)]

    base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44", 
                    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6", 
                    "#FF8C94", "#86A8E7", "#D4A5A5", "#9ED9CC", "#FFE156", "#8B4513"]
    
    behaviours = combined_metrics['behaviour'].unique()
    behaviours = [b for b in behaviours if b != 'weighted_avg'] # remove average
    
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
        palette=['black'],
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

def plot_auc_comparison(combined_metrics, BASE_PATH):
    """
    Creates a faceted plot comparing AUC scores across datasets and training conditions.
    Datasets are shown as rows and training conditions as columns.
    
    Args:
        base_path (str): Base path where the metrics files are stored
    """
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
    plot_path = Path(f"{BASE_PATH}/Output/Testing/Plots/dataset_comparison_AUC.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

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
    save_path = Path(f"{BASE_PATH}/Output/Testing/Plots/{DATASET_NAME}_combined_confusion_matrices.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_set_comparison(combined_metrics, BASE_PATH):    

    # select only the key behaviours
    plot_behaviours = combined_metrics[combined_metrics['model_type'].str.lower() == 'binary']['behaviour'].unique()
    combined_metrics = combined_metrics[combined_metrics['behaviour'].isin(plot_behaviours)]

    base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", 
                    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6"]
    
    behaviours = [b for b in plot_behaviours if b != 'weighted_avg'] # remove average
    
    # Create colour dictionary
    colour_dict = dict(zip(behaviours, base_colors))

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
    training_set_order = ['all', 'some', 'target']
    
    # Create faceted plot with datasets as rows and model_type as columns
    g = sns.FacetGrid(melted_metrics, 
                      row='dataset',
                      col='model_type',
                      height=4,
                      aspect=0.75)
    
    # Draw points for weighted_avg data using default rainbow colors
    g.map_dataframe(lambda data, **kwargs: sns.scatterplot(
        data=data[data['behaviour'] != 'weighted_avg'].assign(
            model_type=lambda x: pd.Categorical(x['model_type'], categories=model_order, ordered=True)
        ),
        x='training_set',
        y='value',
        hue='behaviour',
        palette=colour_dict,  # Using the custom colour dictionary
        s=100))
    
    # Draw stars for weighted_avg data
    g.map_dataframe(lambda data, **kwargs: sns.scatterplot(
        data=data[data['behaviour'] == 'weighted_avg'].assign(
            training_set=lambda x: pd.Categorical(x['training_set'], categories=training_set_order, ordered=True)
        ),
        x='training_set',
        y='value',
        hue='behaviour',
        palette=['black'],  # All weighted averages set to black
        s=200,
        marker='*'))
    
    # Customize plot appearance
    for ax in g.axes.flat:
        # Set y-axis limits to show full range from 0 to 1
        ax.set_ylim(0.4, 1.0)
        
        # Add y-ticks at regular intervals
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelrotation=45)
    
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.map(plt.grid, axis='y', linestyle='--', alpha=0.7)
    g.set_axis_labels('Training Set', 'AUC Score')
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    # Adjust layout and save as a PDF
    plt.tight_layout()
    plot_path = Path(f"{BASE_PATH}/Output/Testing/Plots/dataset_comparison_training_set.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def combine_metrics_per_fold(BASE_PATH):
    # for combining all the metrics within each fold
    all_metrics = []
    for fold in [1,2,3]:
        # read all files in the the metrics folder
        folder_path = Path(f"{BASE_PATH}/Output/fold_{fold}/Testing/Metrics/")
        metrics_files = list(folder_path.glob('*.csv'))

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
                if "Activity" in filename_parts[4]:
                    metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}_{filename_parts[5]}"
                    if "fullclasses" in filename_parts[6]:
                        metrics['closed_open'] = f"{filename_parts[6]}_{filename_parts[7].replace('_metrics.csv', '')}"
                if "Other" in filename_parts[4]:
                    metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}"

        # name the first column
        metrics = metrics.rename(columns={metrics.columns[0]: 'behaviour'})
        
        all_metrics.append(metrics)
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # save it to file
        save_path = Path(f"{BASE_PATH}/Output/fold_{fold}/Testing/Metrics/{DATASET_NAME}_all_metrics.csv")
        combined_metrics.to_csv(save_path, index=False)

def combine_metrics_across_folds(BASE_PATH):
    # for combining all the metrics across all folds
    all_metrics = []
    for fold in [1,2,3]:
        metrics = pd.read_csv(f"{BASE_PATH}/Output/fold_{fold}/Testing/{DATASET_NAME}_all_metrics.csv")
        metrics['fold'] = fold
        all_metrics.append(metrics)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # #combine all the metrics
    save_path = Path(f"{BASE_PATH}/Output/Combined/all_combined_metrics.csv")
    combined_metrics.to_csv(save_path, index=False)

def average_metrics_across_folds(BASE_PATH):
    # for averaging the metrics across all folds
    combined_metrics = pd.read_csv(Path(f"{BASE_PATH}/Output/Combined/all_combined_metrics.csv"))
    
    # average the metrics across all folds, save mean and std
    averaged_metrics = combined_metrics.groupby(['dataset', 'model_type', 'training_set', 'behaviour']).agg({'AUC': ['mean', 'std']}).reset_index()

    # save the averaged metrics
    save_path = Path(f"{BASE_PATH}/Output/Combined/all_averaged_metrics.csv")
    averaged_metrics.to_csv(save_path, index=False)

def main(BASE_PATH, target_activities):
    combine_metrics_per_fold(BASE_PATH)
    combine_metrics_across_folds(BASE_PATH)
    average_metrics_across_folds(BASE_PATH)

    # now generate the plots  
    print("beginning plots")
    # for DATASET_NAME in ['Vehkaoja_Dog', 'Ferdinandy_Dog']:
    #     TARGET_ACTIVITIES = target_activities[DATASET_NAME]
    #     plot_model_type_comparison(combined_metrics, DATASET_NAME, TARGET_ACTIVITIES, BASE_PATH)
    #     combine_confusion_matrices(BASE_PATH, DATASET_NAME)

    averaged_metrics = pd.read_csv(Path(f"{BASE_PATH}/Output/Combined/all_averaged_metrics.csv"))
    # plot_auc_comparison(averaged_metrics, BASE_PATH)
    plot_training_set_comparison(averaged_metrics, BASE_PATH)

    print("done")

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES)
