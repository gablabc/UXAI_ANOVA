"""
This script searches for hyperparameters that lead to good 
tree-based models such as Random Forests and Gradient Boosted Trees. 
The average error of the model across several train/valid split 
is logged on W&B in order to guide the hyperparameter search.

It is important to note that the random sampling of hyperparameters
is done inside the grid provided at 
models/sweeps/<model_name>_<data_name>.json. This file can be 
changed in order to refine the search.

Example usage with command line:

    To see available arguments:
        python tune_tree_ensemble.py --help

    Example command
        python tune_tree_ensemble.py --name=bike --model_name=rf
"""

import wandb
import os

# Sklearn
from sklearn.model_selection import RandomizedSearchCV

# Local
from utils import Wandb_Config, Data_Config, Search_Config, setup_data_trees
from utils import get_cross_validator, get_hp_grid, custom_train_test_split
from utils import TREES



if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    ################################### Setup #################################
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Wandb_Config, "wandb")
    parser.add_arguments(Data_Config, "data")
    parser.add_arguments(Search_Config, "search")
    parser.add_argument("--model_name", type=str, default="rf", 
                        help="Type of tree ensemble either gbt or rf")
    
    args, unknown = parser.parse_known_args()
    print(args)

    X, y, features, task, ohe = setup_data_trees(args.data.name)
    # Encode for training
    if ohe is not None:
        X = ohe.transform(X)
    x_train, _, y_train, _ = custom_train_test_split(X, y.ravel(), task)
    
    # Hyperparameter grid
    hp_grid = get_hp_grid(os.path.join("models", "sweeps", 
                              f"{args.model_name}_{args.data.name}_grid.json"))
    
    # Cross validator for train/valid splits
    cross_validator = get_cross_validator(args.search.n_splits, task, 
                                          args.search.split_seed, 
                                          args.search.split_type)
    
    # Load rf or gbt
    model = TREES[args.model_name][task]
    # model.set_params(n_jobs=-1)
    
    scoring = "neg_root_mean_squared_error" if task == "regression" else "accuracy"
    
    n_repetitions = 20
    cv_search = RandomizedSearchCV(model, hp_grid, scoring=scoring,
                                    cv=cross_validator, n_iter=n_repetitions,
                                    verbose=2).fit(x_train, y_train)
    
    if args.wandb.wandb:
        results = cv_search.cv_results_
        for run_idx in range(n_repetitions):
            run = wandb.init(reinit=True, 
                              project=f"UXAI_{args.model_name}_{args.data.name}", 
                                                                entity="galabc")
            with run:
                run.config.update(results['params'][run_idx])
                run.summary["mean_score"] = results["mean_test_score"][run_idx]
                run.summary["std_score"] = results["std_test_score"][run_idx]

