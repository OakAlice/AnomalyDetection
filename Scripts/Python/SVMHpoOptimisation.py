import os
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import time
from functions.svm_model_tuning import dichotomousModelTuningSVM, multiclassModelTuningSVM
from functions.other_functions import save_best_params, save_results

bounds = {
    'nu': (0.01, 0.5),
    'gamma': (0.001, 1),
    'kernel': (1, 3),
    'number_features': (5, 100)
}

for training_set in TRAINING_SETS:
    try:
        # Dichotomous Model tuning master call
        model_types = ["OCC", "Binary"]
        
        feature_data = pd.read_csv(
            os.path.join(base_path, "Data", "Feature_data", 
                        f"{dataset_name}_other_features.csv")
        ).drop(['GeneralisedActivity', 'OtherActivity'], axis=1)

        results_stored = {}

        for model in model_types:
            for activity in target_activities:
                try:
                    print(f"Tuning {model} model for activity: {activity} from set {training_set}")

                    # Perform Bayesian optimization
                    start_time = time.time()
                    optimizer = BayesianOptimization(
                        f=lambda nu, gamma, kernel, number_features: dichotomousModelTuningSVM(
                            model=model,
                            activity=activity,
                            feature_data=feature_data,
                            nu=nu,
                            kernel=int(kernel),
                            gamma=gamma,
                            number_features=int(number_features),
                            validation_proportion=VALIDATION_PROPORTION,
                            balance=BALANCE
                        ),
                        pbounds=bounds,
                        random_state=1
                    )
                    
                    optimizer.maximize(
                        init_points=5,
                        n_iter=10,
                        kappa=2.576
                    )
                    elapsed_time = time.time() - start_time

                    # Store results for this activity
                    result = save_best_params(
                        data_name=dataset_name,
                        model_type=model,
                        activity=activity,
                        elapsed_time=elapsed_time,
                        results=optimizer
                    )
                    
                    if result is not None:
                        results_stored[activity] = result
                    else:
                        print(f"Skipping activity {activity} due to error.")

                except Exception as e:
                    print(f"Error processing activity: {activity} for model: {model}\nError message: {str(e)}")
                    continue

            save_results(
                results_stored,
                os.path.join(base_path, "Output", "Tuning", ML_METHOD,
                            f"{dataset_name}_{training_set}_{model}_hyperparmaters.csv")
            )

        # Multiclass model tuning
        behaviour_columns = ["Activity", "OtherActivity", "GeneralisedActivity"]

        # Load feature data for multiclass optimization
        feature_data = pd.read_csv(
            os.path.join(base_path, "Data", "Feature_data",
                        f"{dataset_name}_other_features.csv")
        )

        results_stored = {}

        for behaviours in behaviour_columns:
            try:
                print(f"Tuning multiclass model for behaviour column: {behaviours} from dataset {training_set}")

                # Prepare data for current grouping
                multiclass_data = feature_data.drop(
                    [col for col in behaviour_columns if col != behaviours], 
                    axis=1
                ).rename(columns={behaviours: "Activity"})

                if behaviours == "GeneralisedActivity":
                    multiclass_data = multiclass_data[multiclass_data["Activity"] != ""]

                # Perform Bayesian optimization for multiclass model
                start_time = time.time()
                optimizer = BayesianOptimization(
                    f=lambda nu, gamma, kernel, number_features: multiclassModelTuningSVM(
                        model="Multi",
                        multiclass_data=multiclass_data,
                        nu=nu,
                        kernel=int(kernel),
                        gamma=gamma,
                        number_features=int(number_features),
                        validation_proportion=VALIDATION_PROPORTION,
                        balance=BALANCE,
                        loops=1
                    ),
                    pbounds=bounds,
                    random_state=1
                )

                optimizer.maximize(
                    init_points=5,
                    n_iter=10,
                    kappa=2.576
                )
                elapsed_time = time.time() - start_time

                results_stored[behaviours] = save_best_params(
                    data_name=dataset_name,
                    model_type="Multi",
                    activity=behaviours,
                    elapsed_time=elapsed_time,
                    results=optimizer
                )

            except Exception as e:
                print(f"Error processing behaviour: {behaviours}\nError message: {str(e)}")
                print(optimizer.res)
                continue

        # Save all multiclass results if we have any
        if results_stored:
            results_df = pd.DataFrame(results_stored).T
            results_df.to_csv(
                os.path.join(base_path, "Output", "Tuning",
                            f"{dataset_name}_{training_set}_Multi_hyperparmaters.csv")
            )

    except Exception as e:
        print(f"Error processing training set: {training_set}\nError message: {str(e)}")
        continue
