
import os, sys
import numpy as np

# Local imports
from utils import setup_data_trees, custom_train_test_split, get_background
from utils import load_trees, load_FDTree, Data_Config, TreeEnsembleHP
from data_utils import INTERACTIONS_MAPPING

sys.path.append(os.path.abspath(".."))
from src.anova import get_ANOVA_1_tree, interventional_treeshap
from src.anova_tree import Partition

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_arguments(Partition, "partition")
    parser.add_arguments(TreeEnsembleHP, "ensemble")
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--background_size", type=int, default=600, 
                       help="Size of the background data")
    args, unknown = parser.parse_known_args()
    print(args)

    # Random state used for fitting
    state = str(args.ensemble.random_state)

    # Make folder for dataset models
    path = os.path.join("models", args.data.name, args.model_name + "_" + state)

    # Load data and model
    X, y, features, task = setup_data_trees(args.data.name)
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
    # Load models
    model, perfs = load_trees(args.data.name, args.model_name, args.ensemble.random_state)
    interactions = INTERACTIONS_MAPPING[args.data.name]

    # Background data
    background = get_background(x_train, args.background_size, args.ensemble.random_state)

    # Do not recompute the shapley values if they were computed
    if not os.path.exists(os.path.join(path, f"phis_global_N_{args.background_size}.npy")):
        phis, _ = interventional_treeshap(model, background, background)
        np.save(os.path.join(path, f"phis_global_N_{args.background_size}.npy"), phis)

    # For FD-Trees fo increasing depths
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(max_depth, args.data.name, args.model_name, args.ensemble.random_state, 
                           args.partition.type, args.background_size)
        groups, rules = tree.predict(background[:, interactions])

        # Explain in each Region
        backgrounds = [0] * tree.n_groups
        for group_idx in range(tree.n_groups):
            idx_select = (groups == group_idx)
            regional_background = background[idx_select]

            # SHAP
            phis, _ = interventional_treeshap(model,
                                            regional_background, 
                                            regional_background)
            filename = f"phis_{args.partition.type}_N_{args.background_size}_" +\
                       f"max_depth_{max_depth}_region_{group_idx}.npy"
            np.save(os.path.join(path, filename), phis)
