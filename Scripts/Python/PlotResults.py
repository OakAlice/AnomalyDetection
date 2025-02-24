# Script to compare the performance of the models for different conditions
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from MainScript import BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES

def melt_grouped_metrics(combined_metrics, metrics_to_plot):
    
    # First, compute mean and std for all metrics of interest.
    grouped_metrics = combined_metrics.groupby(
        ['dataset', 'model_type', 'training_set', 'behaviour']
    )[metrics_to_plot].agg(['mean', 'std']).reset_index()

    # Flatten the MultiIndex columns (e.g. convert ('F1', 'mean') to 'F1_mean')
    grouped_metrics.columns = ['_'.join(col).strip('_') for col in grouped_metrics.columns.values]

    # Now, for each metric get its mean values with one melt...
    mean_melted = pd.melt(
        grouped_metrics,
        id_vars=['dataset', 'model_type', 'training_set', 'behaviour'],
        value_vars=[f"{metric}_mean" for metric in metrics_to_plot],
        var_name='metric',
        value_name='mean'
    )
    # Remove the trailing '_mean' so that the metric name is clear
    mean_melted['metric'] = mean_melted['metric'].str.replace("_mean", "", regex=False)

    # And separately melt the std values
    std_melted = pd.melt(
        grouped_metrics,
        id_vars=['dataset', 'model_type', 'training_set', 'behaviour'],
        value_vars=[f"{metric}_std" for metric in metrics_to_plot],
        var_name='metric',
        value_name='std'
    )
    std_melted['metric'] = std_melted['metric'].str.replace("_std", "", regex=False)

    # Merge the mean and std results matching on all identifier columns and the metric
    melted_metrics = pd.merge(
        mean_melted,
        std_melted,
        on=['dataset', 'model_type', 'training_set', 'behaviour', 'metric']
    )

    if melted_metrics.empty:
        print("Warning: melted_metrics is empty. Check your filtering conditions.")
        return

    print("Original dataset behaviours:", combined_metrics['behaviour'].unique())
    melted_metrics = melt_grouped_metrics(combined_metrics, metrics_to_plot)
    print("Melted metrics behaviours:", melted_metrics['behaviour'].unique())

    return melted_metrics


def plot_model_type_comparison(combined_metrics, BASE_PATH):
    for DATASET_NAME in combined_metrics['dataset'].unique():
        # Create a subset of the metrics for the current dataset
        dataset_metrics = combined_metrics[combined_metrics['dataset'] == DATASET_NAME]
        
        # For plotting, allow weighted_avg passes through by including it explicitly:
        plot_behaviours = np.append(
            dataset_metrics[dataset_metrics['model_type'].str.lower() == 'binary']['behaviour'].unique(),
            'weighted_avg'
        )
        dataset_metrics = dataset_metrics[dataset_metrics['behaviour'].isin(plot_behaviours)]
        dataset_metrics = dataset_metrics[dataset_metrics['closed_open'] == 'open']

        metrics_to_plot = ['F1', 'AUC', 'Precision', 'Recall']
        
        # Calculate mean and std for each group
        melted_metrics = []
        for metric in metrics_to_plot:
            grouped = dataset_metrics.groupby(
                ['training_set', 'model_type', 'behaviour']
            )[metric].agg(['mean', 'std']).reset_index()
            grouped['metric'] = metric
            melted_metrics.append(grouped)
        
        melted_metrics = pd.concat(melted_metrics, ignore_index=True)
        
        # Check if melted_metrics is empty
        if melted_metrics.empty:
            print(f"No melted metrics for dataset {DATASET_NAME}. Skipping plot_model_type_comparison for this dataset.")
            continue
        
        base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44", 
                       "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6", 
                       "#FF8C94", "#86A8E7", "#D4A5A5", "#9ED9CC", "#FFE156", "#8B4513"]
        
        behaviours = [b for b in plot_behaviours if b != 'weighted_avg']  # remove average
        
        # Create colour dictionary
        colour_dict = dict(zip(behaviours, base_colors[:len(behaviours)]))

        # Define the order for model_type
        model_order = ['multi_Activity_NOthreshold', 'oneclass', 'binary', 'multi_Other', 
                       'multi_Activity_threshold']
        
        # Convert model_type to categorical with specified order
        melted_metrics['model_type'] = pd.Categorical(
            melted_metrics['model_type'],
            categories=model_order,
            ordered=True
        )
        
        # Create faceted plot
        g = sns.FacetGrid(melted_metrics, 
                         col='training_set',
                         row='metric',
                         height=4,
                         aspect=1.5)
        
        # Create empty lists to store legend elements
        legend_elements = []
        legend_labels = []
        
        # Draw points and error bars
        def plot_with_errorbars(data, **kwargs):
            ax = plt.gca()
            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels(model_order)
            
            # Plot non-weighted behaviors
            non_weighted = data[data['behaviour'] != 'weighted_avg']
            for behavior in non_weighted['behaviour'].unique():
                behavior_data = non_weighted[non_weighted['behaviour'] == behavior]
                x_positions = [model_order.index(mt) for mt in behavior_data['model_type']]
                line = plt.errorbar(
                    x_positions, 
                    behavior_data['mean'],
                    yerr=behavior_data['std'],
                    fmt='o',
                    color=colour_dict.get(behavior),
                    alpha=0.7,
                    ecolor=colour_dict.get(behavior),
                    elinewidth=1.5,
                    capsize=5,
                    capthick=1,
                    markersize=8,
                    label=behavior
                )
                # Only add to legend elements if not already present
                if behavior not in legend_labels:
                    legend_elements.append(line)
                    legend_labels.append(behavior)
            
            # Plot weighted average
            weighted_data = data[data['behaviour'] == 'weighted_avg']
            if not weighted_data.empty:
                x_positions = [model_order.index(mt) for mt in weighted_data['model_type']]
                line = plt.errorbar(
                    x_positions,
                    weighted_data['mean'],
                    yerr=weighted_data['std'],
                    fmt='*',
                    color='black',
                    alpha=0.7,
                    ecolor='black',
                    elinewidth=1.5,
                    capsize=5,
                    capthick=1,
                    markersize=12,
                    label='weighted_avg'
                )
                # Only add weighted_avg to legend once
                if 'weighted_avg' not in legend_labels:
                    legend_elements.append(line)
                    legend_labels.append('weighted_avg')

        g.map_dataframe(plot_with_errorbars)
        
        # Customize plot appearance
        for ax in g.axes.flat:
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='x', labelrotation=45)
            # Remove individual legends
            ax.get_legend().remove() if ax.get_legend() is not None else None
        
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.map(plt.grid, axis='y', linestyle='--', alpha=0.7)
        g.set_axis_labels('Model Type', 'Value')
        
        # Add a single legend to the figure
        g.fig.legend(legend_elements,
                    legend_labels,
                    title='Behaviour',
                    bbox_to_anchor=(1.05, 0.5),
                    loc='center left',
                    borderaxespad=0.)
        
        plt.tight_layout()

        # Save the plot
        plot_path = Path(f"{BASE_PATH}/Output/Combined/{DATASET_NAME}_model_type_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()





def plot_training_set_comparison(combined_metrics, BASE_PATH):    
    # Select only key behaviours
    plot_behaviours = combined_metrics[combined_metrics['model_type'].str.lower() == 'binary']['behaviour'].unique()
    combined_metrics = combined_metrics[combined_metrics['behaviour'].isin(plot_behaviours)]

    base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", 
                   "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6"]
    
    # Remove the closed condition
    combined_metrics = combined_metrics[combined_metrics['closed_open'] == 'open']
    
    behaviours = [b for b in plot_behaviours if b != 'weighted_avg']
    
    # Create colour dictionary
    colour_dict = dict(zip(behaviours, base_colors))

    # Define order for model_type and training_set
    model_order = ['multi_Activity_NOthreshold', 'oneclass', 'binary', 'multi_Other', 'multi_Activity_threshold']
    training_set_order = ['all', 'some', 'target']

    # Calculate mean and std for each group
    grouped_metrics = combined_metrics.groupby(
        ['dataset', 'model_type', 'training_set', 'behaviour']
    )['AUC'].agg(['mean', 'std']).reset_index()

    # Convert to categorical with specified order
    grouped_metrics['model_type'] = pd.Categorical(grouped_metrics['model_type'], categories=model_order, ordered=True)
    grouped_metrics['training_set'] = pd.Categorical(grouped_metrics['training_set'], categories=training_set_order, ordered=True)
    
    # Set aesthetic style (no grid)
    sns.set_style('white')

    # Create faceted plot with taller & narrower aspect ratio
    g = sns.FacetGrid(grouped_metrics, 
                      row='dataset',
                      col='model_type',
                      height=5,  # Increase height
                      aspect=0.4,  # Decrease aspect for narrow plots
                      col_order=model_order)
    
    # Draw points and error bars using the std column for error bars
    def plot_with_errorbars(data, **kwargs):
        ax = plt.gca()
        ax.set_xticks(range(len(training_set_order)))
        ax.set_xticklabels(training_set_order)
        
        for behavior in data['behaviour'].unique():
            behavior_data = data[data['behaviour'] == behavior]
            x_positions = [training_set_order.index(ts) for ts in behavior_data['training_set']]
            plt.errorbar(
                x_positions, 
                behavior_data['mean'],          # Use the mean value for the y-axis
                yerr=behavior_data['std'],      # Standard deviation used as error bars
                fmt='o',
                color=colour_dict.get(behavior),
                alpha=0.7,                      # Semi-transparent markers
                ecolor=colour_dict.get(behavior),
                elinewidth=1.5,
                capsize=5,
                capthick=1,
                markersize=8,
                label=behavior
            )

    g.map_dataframe(plot_with_errorbars)
    
    # Draw stars for weighted_avg data
    def plot_weighted_avg(data, **kwargs):
        weighted_data = data[data['behaviour'] == 'weighted_avg']
        if not weighted_data.empty:
            x_positions = [training_set_order.index(ts) for ts in weighted_data['training_set']]
            plt.errorbar(x_positions,
                         weighted_data['mean'],
                         yerr=weighted_data['std'],
                         fmt='*',
                         color='black',
                         alpha=0.7,  # Semi-transparent stars
                         capsize=5,
                         capthick=1,
                         markersize=12,
                         label='weighted_avg')

    g.map_dataframe(plot_weighted_avg)
    
    # Remove grid and set thick black border
    for ax in g.axes.flat:
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticks(range(len(training_set_order)))
        ax.set_xticklabels(training_set_order)

        # Remove grid
        ax.grid(False)

        # Add thick black border to each subplot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

    # Add thick black border around the entire figure
    g.figure.patch.set_linewidth(2)
    g.figure.patch.set_edgecolor('black')

    # Add a global legend outside the grid
    g.set_titles(row_template="{row_name}")
    g.set_titles(col_template="{col_name}") # Add column titles
    g.map(plt.grid, axis='y', linestyle='--', alpha=0.7) # Add y-axis grid lines for better readability
    g.set_axis_labels('Model Type', 'Value') # Add acix and plot labels
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left') # Add a legend outside the plot
    plt.tight_layout() # Adjust layout to prevent label overlap

    # Adjust layout and save
    plt.tight_layout()
    plot_path = Path(f"{BASE_PATH}/Output/Combined/AUC_comparing_training_sets.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def plot_false_positive_rate(combined_metrics, BASE_PATH):
    # Create a subset of the metrics for the current dataset
    dataset_metrics = combined_metrics[combined_metrics['dataset'] == DATASET_NAME]
    # Remove the closed condition
    dataset_metrics = dataset_metrics[dataset_metrics['closed_open'] == 'open']

    # Create colour dictionary
    base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", 
                   "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6"] 
    behaviours = [b for b in dataset_metrics['behaviour'].unique() if b != 'weighted_avg']
    colour_dict = dict(zip(behaviours, base_colors))

    # Define order for model_type and training_set
    model_order = ['multi_Activity_NOthreshold', 'oneclass', 'binary', 'multi_Other', 'multi_Activity_threshold']
    training_set_order = ['all', 'some', 'target']

    # Calculate mean and std for each group
    grouped_metrics = dataset_metrics.groupby(
        ['dataset', 'model_type', 'training_set', 'behaviour']
    )['FPR'].agg(['mean', 'std']).reset_index()

    # Convert to categorical with specified order
    grouped_metrics['model_type'] = pd.Categorical(grouped_metrics['model_type'], categories=model_order, ordered=True)
    grouped_metrics['training_set'] = pd.Categorical(grouped_metrics['training_set'], categories=training_set_order, ordered=True)
    
    # Set aesthetic style (no grid)
    sns.set_style('white')

    # Create faceted plot with taller & narrower aspect ratio
    g = sns.FacetGrid(grouped_metrics, 
                      col='model_type',
                      height=5,  # Increase height
                      aspect=0.4,  # Decrease aspect for narrow plots
                      col_order=model_order)
    
    # Draw points and error bars using the std column for error bars
    def plot_with_errorbars(data, **kwargs):
        ax = plt.gca()
        ax.set_xticks(range(len(training_set_order)))
        ax.set_xticklabels(training_set_order)
        
        normal_data = data[data['behaviour'] != 'weighted_avg']

        for behaviour in normal_data['behaviour'].unique():
            behaviour_data = normal_data[normal_data['behaviour'] == behaviour]
            x_positions = [training_set_order.index(ts) for ts in behaviour_data['training_set']]
            plt.errorbar(
                x_positions, 
                behaviour_data['mean'],          # Use the mean value for the y-axis
                yerr=behaviour_data['std'],      # Standard deviation used as error bars
                fmt='o',
                color=colour_dict.get(behaviour),
                alpha=0.7,                      # Semi-transparent markers
                ecolor=colour_dict.get(behaviour),
                elinewidth=1.5,
                capsize=5,
                capthick=1,
                markersize=8,
                label=behaviour
            )

    g.map_dataframe(plot_with_errorbars)
    
    # Draw stars for weighted_avg data
    def plot_weighted_avg(data, **kwargs):
        weighted_data = data[data['behaviour'] == 'weighted_avg']
        if not weighted_data.empty:
            x_positions = [training_set_order.index(ts) for ts in weighted_data['training_set']]
            plt.errorbar(x_positions,
                         weighted_data['mean'],
                         yerr=weighted_data['std'],
                         fmt='*',
                         color='black',
                         alpha=0.7,  # Semi-transparent stars
                         capsize=5,
                         capthick=1,
                         markersize=12,
                         label='weighted_avg')

    g.map_dataframe(plot_weighted_avg)
    
    # Remove grid and set thick black border
    for ax in g.axes.flat:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticks(range(len(training_set_order)))
        ax.set_xticklabels(training_set_order)

        # Remove grid
        ax.grid(False)

        # Add thick black border to each subplot
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

    # Add thick black border around the entire figure
    g.figure.patch.set_linewidth(2)
    g.figure.patch.set_edgecolor('black')

    # Add a global legend outside the grid
    g.set_titles(row_template="{row_name}")
    g.set_titles(col_template="{col_name}") # Add column titles
    g.map(plt.grid, axis='y', linestyle='--', alpha=0.7) # Add y-axis grid lines for better readability
    g.set_axis_labels('Model Type', 'Value') # Add acix and plot labels
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left') # Add a legend outside the plot
    plt.tight_layout() # Adjust layout to prevent label overlap

    # Adjust layout and save
    plt.tight_layout()
    plot_path = Path(f"{BASE_PATH}/Output/Combined/FPR_comparing_training_sets.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def plot_training_set_metrics_grid(combined_metrics, BASE_PATH, DATASET_NAME):
    # Filter dataset and remove 'closed' condition
    dataset_metrics = combined_metrics[
        (combined_metrics['dataset'] == DATASET_NAME) & (combined_metrics['closed_open'] == 'open')
    ]
    
    # Define order for model_type and training_set
    model_order = ['multi_Activity_NOthreshold', 'oneclass', 'binary', 'multi_Other', 'multi_Activity_threshold']
    training_set_order = ['all', 'some', 'target']
    metrics = ['AUC', 'Accuracy', 'Specificity','F1']
    
    # Create colour dictionary
    base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", 
                   "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6"]
    behaviours = [b for b in dataset_metrics['behaviour'].unique() if b != 'weighted_avg']
    colour_dict = dict(zip(behaviours, base_colors))
    
    # Melt dataframe for FacetGrid
    melted_metrics = dataset_metrics.melt(
        id_vars=['dataset', 'model_type', 'training_set', 'behaviour'], 
        value_vars=metrics, 
        var_name='metric', 
        value_name='value'
    )
    
    # Compute mean and std for error bars
    grouped_metrics = melted_metrics.groupby(['metric', 'model_type', 'training_set', 'behaviour'])['value'].agg(['mean', 'std']).reset_index()
    
    # Convert to categorical with specified order
    grouped_metrics['model_type'] = pd.Categorical(grouped_metrics['model_type'], categories=model_order, ordered=True)
    grouped_metrics['training_set'] = pd.Categorical(grouped_metrics['training_set'], categories=training_set_order, ordered=True)
    grouped_metrics['metric'] = pd.Categorical(grouped_metrics['metric'], categories=metrics, ordered=True)
    
    # Set aesthetic style (no grid)
    sns.set_style('white')
    
    # Create FacetGrid with rows for metrics and columns for model_type
    g = sns.FacetGrid(grouped_metrics, row='metric', col='model_type', height=4, aspect=0.8, 
                      row_order=metrics, col_order=model_order)
    
    # Define plotting function
    def plot_with_errorbars(data, **kwargs):
        ax = plt.gca()
        ax.set_xticks(range(len(training_set_order)))
        ax.set_xticklabels(training_set_order)
        
        normal_data = data[data['behaviour'] != 'weighted_avg']

        for behaviour in normal_data['behaviour'].unique():
            behaviour_data = normal_data[normal_data['behaviour'] == behaviour]
            x_positions = [training_set_order.index(ts) for ts in behaviour_data['training_set']]
            plt.errorbar(
                x_positions, 
                behaviour_data['mean'], 
                yerr=behaviour_data['std'],
                fmt='o',
                color=colour_dict.get(behaviour),
                alpha=0.7,
                markersize=8,
                capsize=5,
                elinewidth=1.5,
                label=behaviour
            )
    
    g.map_dataframe(plot_with_errorbars)

    # Draw stars for weighted_avg data
    def plot_weighted_avg(data, **kwargs):
        weighted_data = data[data['behaviour'] == 'weighted_avg']
        if not weighted_data.empty:
            x_positions = [training_set_order.index(ts) for ts in weighted_data['training_set']]
            plt.errorbar(x_positions,
                         weighted_data['mean'],
                         yerr=weighted_data['std'],
                         fmt='*',
                         color='black',
                         alpha=0.7,  # Semi-transparent stars
                         capsize=5,
                         capthick=1,
                         markersize=12,
                         label='weighted_avg')

    g.map_dataframe(plot_weighted_avg)
    
    # Formatting
    for ax in g.axes.flat:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
    
    g.set_axis_labels('Training Set', 'Metric Value')
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(f"{BASE_PATH}/Output/Combined/Metrics_comparing_training_sets.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()




def plot_model_type_metrics_grid(combined_metrics, BASE_PATH, DATASET_NAME):
    # Filter dataset and remove 'closed' condition
    dataset_metrics = combined_metrics[
        (combined_metrics['dataset'] == DATASET_NAME) & (combined_metrics['closed_open'] == 'open')
    ]
    
    # Define order for model_type and training_set
    model_order = ['multi_Activity_NOthreshold', 'oneclass', 'binary', 'multi_Other', 'multi_Activity_threshold']
    training_set_order = ['all', 'some', 'target']
    metrics = ['AUC', 'Accuracy', 'Specificity','F1']
    
    # Create colour dictionary
    base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", 
                   "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6"]
    behaviours = [b for b in dataset_metrics['behaviour'].unique() if b != 'weighted_avg']
    colour_dict = dict(zip(behaviours, base_colors))
    
    # Melt dataframe for FacetGrid
    melted_metrics = dataset_metrics.melt(
        id_vars=['dataset', 'model_type', 'training_set', 'behaviour'], 
        value_vars=metrics, 
        var_name='metric', 
        value_name='value'
    )
    
    # Compute mean and std for error bars
    grouped_metrics = melted_metrics.groupby(['metric', 'model_type', 'training_set', 'behaviour'])['value'].agg(['mean', 'std']).reset_index()
    
    # Convert to categorical with specified order
    grouped_metrics['model_type'] = pd.Categorical(grouped_metrics['model_type'], categories=model_order, ordered=True)
    grouped_metrics['training_set'] = pd.Categorical(grouped_metrics['training_set'], categories=training_set_order, ordered=True)
    grouped_metrics['metric'] = pd.Categorical(grouped_metrics['metric'], categories=metrics, ordered=True)
    
    # Set aesthetic style (no grid)
    sns.set_style('white')
    
    # Create FacetGrid with rows for metrics and columns for training_set
    g = sns.FacetGrid(grouped_metrics, 
                      row='metric', 
                      col='training_set',  # Changed from 'model_type'
                      height=4, 
                      aspect=1,
                      row_order=metrics, 
                      col_order=training_set_order)  # Changed from model_order
    
    def plot_with_errorbars(data, **kwargs):
        ax = plt.gca()
        # Set x-ticks for model types
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha='right')
        
        normal_data = data[data['behaviour'] != 'weighted_avg']

        for behaviour in normal_data['behaviour'].unique():
            behaviour_data = normal_data[normal_data['behaviour'] == behaviour]
            x_positions = [model_order.index(mt) for mt in behaviour_data['model_type']]
            plt.errorbar(
                x_positions, 
                behaviour_data['mean'], 
                yerr=behaviour_data['std'],
                fmt='o',
                color=colour_dict.get(behaviour, 'gray'),
                alpha=0.7,
                markersize=8,
                capsize=5,
                elinewidth=1.5,
                label=behaviour
            )

    g.map_dataframe(plot_with_errorbars)

    # Draw stars for weighted_avg data
    def plot_weighted_avg(data, **kwargs):
        weighted_data = data[data['behaviour'] == 'weighted_avg']
        if not weighted_data.empty:
            x_positions = [model_order.index(mt) for mt in weighted_data['model_type']]
            plt.errorbar(x_positions,
                         weighted_data['mean'],
                         yerr=weighted_data['std'],
                         fmt='*',
                         color='black',
                         alpha=0.7,  # Semi-transparent stars
                         capsize=5,
                         capthick=1,
                         markersize=12,
                         label='weighted_avg')

    g.map_dataframe(plot_weighted_avg)
    
    # Formatting
    for ax in g.axes.flat:
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
    
    # Add titles to the facets
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    
    # Rotate x-axis labels
    plt.setp(g.axes.flat, xticks=range(len(model_order)))
    for ax in g.axes.flat:
        ax.set_xticklabels(model_order, rotation=45, ha='right')
    
    g.set_axis_labels('Model Type', 'Metric Value')
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(f"{BASE_PATH}/Output/Combined/Metrics_comparing_model_types.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def main(BASE_PATH, DATASET_NAME):
    combined_metrics = pd.read_csv(Path(f"{BASE_PATH}/Output/Combined/all_combined_metrics.csv"))
    # now generate the plots  
    print("beginning plots")

    combined_metrics = combined_metrics[combined_metrics['dataset'] == DATASET_NAME]

    # plot_model_type_comparison(combined_metrics, BASE_PATH)
    plot_training_set_metrics_grid(combined_metrics, BASE_PATH, DATASET_NAME)
    plot_model_type_metrics_grid(combined_metrics, BASE_PATH, DATASET_NAME)

    print("done")

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES)
