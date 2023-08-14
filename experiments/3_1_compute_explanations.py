
import os, sys
import numpy as np

# Local imports
from utils import setup_data_trees, custom_train_test_split
from utils import load_trees, load_FDTree, Data_Config
from data_utils import INTERACTIONS_MAPPING

sys.path.append(os.path.abspath(".."))
from src.anova import get_ANOVA_1_tree, interventional_treeshap



if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--background_size", type=int, default=500, 
                       help="Size of the background data")
    args, unknown = parser.parse_known_args()
    print(args)

    
    # Make folder for dataset models
    path = os.path.join("models", args.data.name, args.model_name)

    # Load data and model
    X, y, features, task = setup_data_trees(args.data.name)
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
    # Load models
    models, perfs = load_trees(args.data.name, args.model_name)
    interactions = INTERACTIONS_MAPPING[args.data.name]

    # Uniform Background
    background = x_train[:args.background_size]
    A = get_ANOVA_1_tree(background, models[0], task="regression")
    phis, _ = interventional_treeshap(models[0], background, background)
    np.save(os.path.join(path, "A_global.npy"), A)
    np.save(os.path.join(path, "phis_global.npy"), phis)

    # For FD-Trees fo increasing depths
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(args.data.name, args.model_name, max_depth)
        groups, rules = tree.predict(background[:, interactions])

        # Explain in each Region
        backgrounds = [0] * tree.n_groups
        for group_idx in range(tree.n_groups):
            idx_select = (groups == group_idx)
            regional_background = background[idx_select]

            # SHAP
            phis, _ = interventional_treeshap(models[0],
                                            regional_background, 
                                            regional_background)
            filename = f"phis_max_depth_{max_depth}_region_{group_idx}.npy"
            np.save(os.path.join(path,filename), phis)
