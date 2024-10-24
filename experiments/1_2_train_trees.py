"""
This script trains tree-based models, reports their test error and saves them locally
as models/<dataset>/<model_name>_<random_state>/model.joblib

Example usage with command line:

    To see available arguments:
        python 1_2_train_trees.py --help

    Example command:
        python 1_2_train_trees.py --name=bike --size_ensemble=50 --model_name=rf --random_state=0

"""

import pandas as pd
import numpy as np


# Local imports
from utils import TreeEnsembleHP, RandomForestHP, GradientBoostingHP, TREES
from utils import Data_Config, custom_train_test_split, save_tree, setup_data_trees


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_arguments(TreeEnsembleHP, "ensemble")
    parser.add_arguments(RandomForestHP, "rf")
    parser.add_arguments(GradientBoostingHP, "gbt")
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--save", action='store_true', help="Save model locally")

    args, unknown = parser.parse_known_args()
    print(args)

    X, y, features, task = setup_data_trees(args.data.name)
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
    
    # Load rf or gbt
    model = TREES[args.model_name][task]
    # model.set_params(n_jobs=-1)

    # Set params to user-specified values
    if args.ensemble.max_depth == -1:
        args.ensemble.max_depth = None
    if task == "regression" and args.model_name == "rf":
        args.rf.criterion = "squared_error"
    
    # Set the hyperparameters
    model.set_params(**args.ensemble.__dict__)
    model.set_params(**getattr(args, args.model_name).__dict__)
    print(model.get_params())
    
    # # Train
    model.fit(x_train, y_train)

    # # Assess performance
    if task == "regression":
        perf_train = np.sqrt(np.mean((model.predict(x_train) - y_train) ** 2))
        perf_test  = np.sqrt(np.mean((model.predict(x_test) - y_test) ** 2))
    else:
        perf_train = np.mean(model.predict(x_train) == y_train)
        perf_test  = np.mean(model.predict(x_test) == y_test)
    
    perf_df = pd.DataFrame([[perf_train, perf_test]], columns=["Train", "Test"])
    print(perf_df)
    
    
    if args.save:
        print("Saving Results")
        save_tree(model, args.data.name, args.model_name, args.ensemble.random_state, perf_df)