
import os, sys
import numpy as np

# Local imports
from utils import setup_data_trees, custom_train_test_split
from utils import load_trees, save_FDTree, Data_Config
from data_utils import INTERACTIONS_MAPPING

sys.path.append(os.path.abspath(".."))
from src.anova_tree import FDTree
from src.anova import get_A_treeshap


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--background_size", type=int, default=500,
                       help="Size of the background data")
    parser.add_argument("--save", action='store_true', help="Save model locally")
    args, unknown = parser.parse_known_args()
    print(args)

    # Load data and model
    X, y, features, task = setup_data_trees(args.data.name)
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
    # Load models
    models, perfs = load_trees(args.data.name, args.model_name)

    # Uniform Background TODO This depends on the dataset
    background = x_train[:args.background_size]
    A = get_A_treeshap(models[0], background)

    # Only use interacting features when fitting the FDTree
    interactions = INTERACTIONS_MAPPING[args.data.name]
    subset_features = features.select(interactions)

    for max_depth in [1, 2, 3]:
        # Fit the tree
        tree = FDTree(subset_features, max_depth=max_depth, save_losses=True, 
                        negligible_impurity=0.02*y.var(), relative_decrease=0.9)
        tree.fit(background[:, interactions], A)
        tree.print(verbose=False)
        groups, rules = tree.predict(X)
        print(rules)

        if args.save:
            print("Saving Results")
            save_FDTree(tree, args.data.name, args.model_name)


# # %%
# # Plot the objective values w.r.t the split candidates
# plt.figure()
# for i in range(3):
#     splits = tree.root.splits[i]
#     objectives = tree.root.objectives[i]
#     plt.plot(splits, objectives, '-o', label=features.names[i])
# plt.ylim(0, y.var())
# plt.xlabel(f"Split value")
# plt.ylabel(r"$L_2$ Cost of Exclusion")
# plt.legend()
# plt.show()