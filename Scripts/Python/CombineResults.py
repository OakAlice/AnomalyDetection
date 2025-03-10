import pandas as pd
from pathlib import Path

def combine_metrics_per_fold(BASE_PATH):
    # for combining all the metrics within each fold
    for fold in [1, 2, 3, 4, 5]:
        # Initialize a new list for this fold to avoid accumulating previous folds' data.
        metrics_for_fold = []
        
        # read all files in the metrics folder for the current fold
        folder_path = Path(f"{BASE_PATH}/Output/fold_{fold}/Testing/Metrics/")
        metrics_files = list(folder_path.glob('*.csv'))

        for file in metrics_files:
            metrics = pd.read_csv(file)
            # Add filename as a column
            metrics['file'] = file.name

            # extract the components and make them into columns
            filename_parts = file.name.split('_')

            # Join the first two parts to create the dataset name.
            metrics['dataset'] = '_'.join(filename_parts[:2])
            # The third part gives the training set
            metrics['training_set'] = filename_parts[2]

            # For non-'multi', model_type is simply the 4th token.
            # For multi files, need to combine specifics.
            if filename_parts[3].lower() == "multi":
                if filename_parts[4] == "Activity":
                    metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}_{filename_parts[5]}"
                    # also check whether a closer or open test
                    if len(filename_parts) > 6 and "fullclasses" in filename_parts[6]:
                        metrics['closed_open'] = f"{filename_parts[6]}_{filename_parts[7].replace('metrics.csv', '').replace('_', '')}"
                elif filename_parts[4] == "Other":
                    metrics['model_type'] = f"{filename_parts[3]}_{filename_parts[4]}"
                else:
                    metrics['model_type'] = filename_parts[3]
            else:
                metrics['model_type'] = filename_parts[3]

            # Rename the first column as 'behaviour'
            metrics = metrics.rename(columns={metrics.columns[0]: 'behaviour'})

            metrics_for_fold.append(metrics)

        # Combine the metrics for the current fold only.
        combined_metrics = pd.concat(metrics_for_fold, ignore_index=True)

        # Save the combined metrics for this fold.
        save_path = Path(f"{BASE_PATH}/Output/fold_{fold}/Testing/all_metrics.csv")
        combined_metrics.to_csv(save_path, index=False)

def combine_metrics_across_folds(BASE_PATH):
    # Combine metrics from all folds
    all_metrics = []
    for fold in range(1, 6):
        metrics = pd.read_csv(f"{BASE_PATH}/Output/fold_{fold}/Testing/all_metrics.csv")
        metrics['fold'] = fold
        all_metrics.append(metrics)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # Save the combined metrics
    save_path = Path(f"{BASE_PATH}/Output/Combined/all_combined_metrics.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined_metrics.to_csv(save_path, index=False)
    
    return combined_metrics


def main(BASE_PATH):
    print("combining metrics per fold")
    combine_metrics_per_fold(BASE_PATH)
    print("combining metrics across folds")
    combine_metrics_across_folds(BASE_PATH)

if __name__ == "__main__":
    main(BASE_PATH)

