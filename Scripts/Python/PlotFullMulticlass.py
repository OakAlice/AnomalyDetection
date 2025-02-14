import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import TestModelOpen
import numpy as np
import seaborn as sns

def generate_full_class_data(BASE_PATH, target_activities):
    # Load in the test data
    for DATASET_NAME in ['Vehkaoja_Dog', "Ferdinandy_Dog"]:
        TARGET_ACTIVITIES = target_activities[DATASET_NAME]
        for TRAINING_SET in ['all', 'some', 'target']:
            MODEL_TYPE ='multi'
            BEHAVIOUR_SET = 'Activity'
            THRESHOLDING = False
            TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                            TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS=False)

def generate_false_positive_plots(BASE_PATH):
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


def generate_false_positive_plots_grid(BASE_PATH):
    datasets = ['Vehkaoja_Dog', 'Ferdinandy_Dog']
    training_sets = ['all', 'some', 'target']
    MODEL_TYPE = 'multi'
    BEHAVIOUR_SET = 'Activity'

    # Create a grid with one row per dataset and one column per training set.
    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(training_sets), figsize=(18, 10))  # No shared y-axis

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
        for j, training_set in enumerate(training_sets):
            path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/"
                        f"{dataset}_{training_set}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_fullclasses_confusion_matrix.csv")
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
    output_path = Path(f"{BASE_PATH}/Output/Testing/Plots/FalsePositivePlots.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved as {output_path}")

    plt.show()

def generate_false_positive_plots_red(BASE_PATH):
    datasets = ['Vehkaoja_Dog', 'Ferdinandy_Dog']
    training_sets = ['all', 'some', 'target']
    MODEL_TYPE = 'multi'
    BEHAVIOUR_SET = 'Activity'

    # Create a grid with one row per dataset and one column per training set.
    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(training_sets), figsize=(18, 10))

    # Ensure axes is a 2D array for consistency
    if len(datasets) == 1 and len(training_sets) == 1:
        axes = np.array([[axes]])
    elif len(datasets) == 1 or len(training_sets) == 1:
        axes = np.array(axes).reshape(len(datasets), len(training_sets))

    # Loop over each dataset (row) and training set (column)
    for i, dataset in enumerate(datasets):
        for j, training_set in enumerate(training_sets):
            path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/"
                        f"{dataset}_{training_set}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_fullclasses_confusion_matrix.csv")
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
    output_path = Path(f"{BASE_PATH}/Output/Testing/Plots/FalsePositivePlots_red.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved as {output_path}")

    plt.show()




def main(BASE_PATH, target_activities):
    # generate_full_class_data(BASE_PATH, target_activities)
    generate_false_positive_plots_grid(BASE_PATH)
    generate_false_positive_plots_red(BASE_PATH)

if __name__ == "__main__":
    main(BASE_PATH, target_activities)

