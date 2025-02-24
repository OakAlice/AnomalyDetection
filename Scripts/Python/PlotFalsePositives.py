import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import TestModelOpen
import numpy as np
import seaborn as sns

def generate_full_class_data(BASE_PATH, target_activities, FOLD):
    # Load in the test data
    DATASET_NAME = "Ferdinandy_Dog"
    TARGET_ACTIVITIES = target_activities[DATASET_NAME]
    for TRAINING_SET in ['all', 'some', 'target']:
        for MODEL_TYPE in ['multi', 'binary', 'oneclass']:
            BEHAVIOUR_SET = 'Activity'
            THRESHOLDING = False
            # set the REASSIGN_LABELS to False for the full open set
            TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                        TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS=False, FOLD = FOLD)

def generate_false_positive_plots(BASE_PATH, MODEL_TYPE):
    # plot the number of false positives for each true class
    for DATASET_NAME in ['Vehkaoja_Dog', "Ferdinandy_Dog"]:
        for TRAINING_SET in ['all', 'some', 'target']:
            MODEL_TYPE ='multi'
            BEHAVIOUR_SET = 'Activity'
            THRESHOLDING = False

            # read in the data
            path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_fullclasses_metrics.csv")
            df = pd.read_csv(path)

            # rearrange the dataframe to make it long
            df = df.melt(id_vars=['True_Label', 'Predicted_Label'], var_name='Class', value_name='Count')

            # plot number of predicted_label for each true_label
            plt.figure(figsize=(10, 6))
            plt.bar(df['True_Label'], df['Count'])
            plt.xlabel('True Class')
            plt.ylabel('Number of Predicted Class')
            plt.title(f'Predicted Class for Each True Class - {DATASET_NAME} {TRAINING_SET}')
            plt.show()


def generate_false_positive_plots_grid(BASE_PATH, FOLD, MODEL_TYPE):
    datasets = ['Ferdinandy_Dog']
    training_sets = ['all', 'some', 'target']

    # Create a grid with one row per dataset and one column per training set.
    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(training_sets), figsize=(20, 8))  # No shared y-axis

    if len(datasets) == 1 and len(training_sets) == 1:
        axes = np.array([[axes]])  # Ensure 2D array for consistency
    elif len(datasets) == 1 or len(training_sets) == 1:
        axes = np.array(axes).reshape(len(datasets), len(training_sets))  # Reshape 1D to 2D

    # Loop over each dataset (row)
    for i, dataset in enumerate(datasets):
        legend_handles = []
        legend_labels = []
        color_map = {}  # To store colors for each predicted class (separately per dataset)

        # Use a more aesthetic color palette
        # palette = sns.color_palette("Set2", n_colors=10)
        palette = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44", 
                    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6", 
                    "#FF8C94", "#86A8E7", "#D4A5A5", "#9ED9CC", "#FFE156", "#8B4513"]

        # Loop over each training set (column)
        # pull from the first fold
        for j, training_set in enumerate(training_sets):

            if MODEL_TYPE == 'multi':
                path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/"
                            f"{dataset}_{training_set}_multi_Activity_NOthreshold_fullclasses_confusion_matrix.csv")
            else:
                path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/"
                            f"{dataset}_{training_set}_{MODEL_TYPE}_fullclasses_confusion_matrix.csv")
                
            print(f"Loading: {path}")

            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                print(f"File not found: {path}")
                continue

            # Rename the first column for clarity
            df.rename(columns={'Unnamed: 0': 'True_Label'}, inplace=True)

            # Ensure 'True_Label' is set as index and fetch predicted classes
            df.set_index('True_Label', inplace=True)
            predicted_labels = df.columns  # Predicted class labels
            true_classes = df.index  # True class labels (dataset-specific)

            # Assign colors consistently within this dataset
            if not color_map:
                color_map = {label: palette[i % len(palette)] for i, label in enumerate(predicted_labels)}

            # Access the appropriate subplot axis
            ax = axes[i, j]

            # Create stacked bars
            bottom = np.zeros(len(df))
            for predicted_label in predicted_labels:
                counts = df[predicted_label].values
                bars = ax.bar(true_classes, counts, bottom=bottom, label=predicted_label, color=color_map[predicted_label])
                bottom += counts  # Update the bottom for stacking

                # Collect legend handles once per dataset (first subplot in row)
                if j == 0:
                    legend_handles.append(bars[0])  # One bar per predicted class
                    legend_labels.append(predicted_label)

            # Formatting
            ax.set_xlabel('True Class')
            ax.set_ylabel('Count')
            ax.set_title(f'{dataset} - {training_set}')
            ax.set_xticks(range(len(true_classes)))
            ax.set_xticklabels(true_classes, rotation=45, ha='right')  # Ensure readable x-axis labels

        # Add a legend to the right of each dataset row
        fig.legend(legend_handles, legend_labels, title=f"Predicted Class ({dataset})",
                   loc="upper right", bbox_to_anchor=(1.05, 1 - (i / len(datasets))), ncol=1)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit legends

    # Save figure as a PDF
    output_path = Path(f"{BASE_PATH}/Output/Combined/Fold_{FOLD}_{MODEL_TYPE}_FalsePositivePlots.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved as {output_path}")

    plt.show()

def generate_false_positive_plots_red(BASE_PATH, FOLD, MODEL_TYPE):
    datasets = ['Ferdinandy_Dog']
    training_sets = ['all', 'some', 'target']

    # Create a grid with one row per dataset and one column per training set.
    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(training_sets), figsize=(20, 8))

    # Ensure axes is a 2D array for consistency
    if len(datasets) == 1 and len(training_sets) == 1:
        axes = np.array([[axes]])
    elif len(datasets) == 1 or len(training_sets) == 1:
        axes = np.array(axes).reshape(len(datasets), len(training_sets))

    # Loop over each dataset (row) and training set (column)
    for i, dataset in enumerate(datasets):
        for j, training_set in enumerate(training_sets):
            if MODEL_TYPE == 'multi':
                path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/"
                            f"{dataset}_{training_set}_multi_Activity_NOthreshold_fullclasses_confusion_matrix.csv")
            else:
                path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/"
                            f"{dataset}_{training_set}_{MODEL_TYPE}_fullclasses_confusion_matrix.csv")
            print(f"Loading: {path}")

            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                print(f"File not found: {path}")
                continue

            # Rename first column for clarity and set it as index
            df.rename(columns={'Unnamed: 0': 'True_Label'}, inplace=True)
            df.set_index('True_Label', inplace=True)

            predicted_labels = df.columns  # Predicted class labels
            true_classes = df.index        # True class labels

            # Normalize each row so that the summed percentage equals 100%
            df = df.div(df.sum(axis=1), axis=0).fillna(0) * 100

            # Compute correct (green) and incorrect (red) proportions for each true class.
            green_values = []
            red_values = []
            for label in true_classes:
                # Get the correct prediction count if available, else 0.
                correct = df.loc[label, label] if label in predicted_labels else 0
                total = df.loc[label].sum()  # Should be 100 after normalization
                green_values.append(correct)
                red_values.append(total - correct)

            # Access the appropriate subplot axis
            ax = axes[i, j]

            # Plot the stacked bars:
            # The bottom (green) part shows correct predictions and the top (red) shows incorrect.
            ax.bar(true_classes, green_values, color="#3BB273", label="Correct")
            ax.bar(true_classes, red_values, bottom=green_values, color="#e08d9e", label="Incorrect")

            # Add axis labels and formatting
            ax.set_xlabel('True Class')
            ax.set_ylabel('Proportion (%)')
            ax.set_title(f'{dataset} - {training_set}')
            ax.set_xticks(range(len(true_classes)))
            ax.set_xticklabels(true_classes, rotation=45, ha='right')
            ax.set_ylim(0, 100)

    # Build a global legend using matplotlib patches.
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#3BB273", label="Correct"),
                       Patch(facecolor="#e08d9e", label="Incorrect")]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.95, 0.95))

    # Adjust layout to account for the legend and save the figure as a PDF.
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    output_path = Path(f"{BASE_PATH}/Output/Combined/Fold_{FOLD}_{MODEL_TYPE}_FalsePositivePlots_red.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved as {output_path}")

    plt.show()



def performance_of_control(BASE_PATH):
    # plot the performance of the control model
    path = Path(f"{BASE_PATH}/Output/Combined/all_combined_metrics.csv")
    df = pd.read_csv(path)

    df_subset = df[(df['model_type'] == 'multi_Activity_NOthreshold') & 
                   (df['dataset'] == 'Ferdinandy_Dog')]
    
    # plot
    plot_behaviours = df_subset[df_subset['closed_open'].str.lower() == 'open']['behaviour'].unique()
    
    base_colors = ["#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", 
                   "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFB347", "#7D5BA6"]
    # Create colour dictionary
    behaviours = [b for b in plot_behaviours if b != 'weighted_avg']
    colour_dict = dict(zip(behaviours, base_colors))

    # Define order
    training_set_order = ['all', 'some', 'target']

    # Calculate mean and std for each group across the folds
    grouped_metrics = df_subset.groupby(
        ['dataset', 'closed_open', 'training_set', 'behaviour']
    )['AUC'].agg(['mean', 'std']).reset_index()

    # Convert to categorical with specified order
    grouped_metrics['training_set'] = pd.Categorical(grouped_metrics['training_set'], categories=training_set_order, ordered=True)
    
    # Set aesthetic style (no grid)
    sns.set_style('white')

    # Create faceted plot
    g = sns.FacetGrid(grouped_metrics, 
                      col='closed_open',
                      height=5,  # Increase height
                      aspect=0.4
                      )
    
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
    g.set_titles(col_template="{col_name}") # Add column titles
    g.map(plt.grid, axis='y', linestyle='--', alpha=0.7) # Add y-axis grid lines for better readability
    g.set_axis_labels('Training Set', 'Value') # Add acix and plot labels
    g.add_legend(title='Behaviour', bbox_to_anchor=(1.05, 0.5), loc='center left') # Add a legend outside the plot
    plt.tight_layout() # Adjust layout to prevent label overlap

    # Adjust layout and save
    plt.tight_layout()
    plot_path = Path(f"{BASE_PATH}/Output/Combined/Comparing_control_conditions.pdf")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()





def main(BASE_PATH, target_activities, FOLD):
    # generate_full_class_data(BASE_PATH, target_activities)

    for MODEL_TYPE in ['multi', 'binary']:
        generate_false_positive_plots_grid(BASE_PATH, FOLD, MODEL_TYPE)
        generate_false_positive_plots_red(BASE_PATH, FOLD, MODEL_TYPE)

    # performance_of_control(BASE_PATH)

if __name__ == "__main__":
    main(BASE_PATH, target_activities, FOLD)

