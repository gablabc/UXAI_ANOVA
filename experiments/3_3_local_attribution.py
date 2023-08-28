
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import plot_legend, attrib_scatter_plot
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_FDTree, get_background
from utils import pdp_vs_shap, Data_Config, TreeEnsembleHP
from data_utils import INTERACTIONS_MAPPING, SCATTER_SHOW

from src.anova_tree import Partition

setup_pyplot_font(25)


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_arguments(Partition, "partition")
    parser.add_arguments(TreeEnsembleHP, "ensemble")
    parser.add_argument("--model_name", type=str, default="gbt", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--background_size", type=int, default=500,
                       help="Size of the background data")
    parser.add_argument("--ncol", type=int, default=2, 
                       help="Number of columns in the Legend")
    parser.add_argument("--save", action='store_true', help="Save disagreement metrics")
    parser.add_argument("--plot", action='store_true', help="Plots the local feature attributions")
    args, unknown = parser.parse_known_args()
    print(args)
    
    # Random state used for fitting
    state = str(args.ensemble.random_state)

    # Make folder for dataset models
    folder_path = os.path.join("Images", args.data.name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    image_path = os.path.join(folder_path, args.model_name + "_" + state)
    # Make folder for architecture
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Load data and model
    X, y, features, task = setup_data_trees(args.data.name)
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
    interactions = INTERACTIONS_MAPPING[args.data.name]

    # Make folder for dataset models
    model_path = os.path.join("models", args.data.name, args.model_name + "_" + state)

    # Get the pre-computed feature attributions
    A = np.load(os.path.join(model_path, f"A_global_N_{args.background_size}.npy"))
    pdp = A[..., 1:].mean(axis=1)
    phis = np.load(os.path.join(model_path, f"phis_global_N_{args.background_size}.npy"))
    background = get_background(x_train, args.background_size, args.ensemble.random_state)

    # Compute disagreement for full background
    pdp_shap_error = [pdp_vs_shap(pdp, phis)]

    # Compare PDP and Shapley Values
    if args.plot:
        for i in SCATTER_SHOW[args.data.name]:
            # For bike we have specific xticks
            attrib_scatter_plot(background, pdp, phis, i, features, args)
            filename = f"Attribution_{i}_N_{args.background_size}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')


    # For various depths of FD-Tree
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(max_depth, args.data.name, args.model_name, args.ensemble.random_state, 
                           args.partition.type, args.background_size)
        groups, rules = tree.predict(background[:, interactions], latex_rules=True)

        # Store the disagreements here
        pdp_shap_error.append(0)

        # Explain in each Region
        phis = [0] * tree.n_groups
        pdps = [0] * tree.n_groups
        backgrounds = [0] * tree.n_groups
        for group_idx in range(tree.n_groups):
            idx_select = (groups == group_idx)
            regional_background = background[idx_select]
            backgrounds[group_idx] = regional_background

            # SHAP
            filename = f"phis_{args.partition.type}_N_{args.background_size}_"+\
                       f"max_depth_{max_depth}_region_{group_idx}.npy"
            phis[group_idx] = np.load(os.path.join(model_path, filename))

            # Reshape idx to index the A matrix
            idx_select = np.where(idx_select)[0].reshape((-1, 1))
            pdps[group_idx] = A[..., 1:][idx_select, idx_select.T].mean(axis=1)

            ########### Local Feature Attribution ############

            # Compute disagreement
            pdp_shap_error[-1] += pdp_vs_shap(pdps[group_idx], phis[group_idx]) * len(idx_select)    
        pdp_shap_error[-1] /= args.background_size


        # Show scatter plots of PDP and SHAP
        if args.plot:
            for i in SCATTER_SHOW[args.data.name]:
                plt.figure()
                attrib_scatter_plot(backgrounds, pdps, phis, i, features, args)
                filename = f"Attribution_{i}_{args.partition.type}_N_{args.background_size}_max_depth_{max_depth}.pdf"
                plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')
        
            # Plot the legend separately
            plot_legend(rules, ncol=args.ncol)
            filename = f"Legend_{args.partition.type}_N_{args.background_size}_max_depth_{max_depth}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight', pad_inches=0)

    for max_depth in [0, 1, 2, 3]:
        print(f"PDP vs SHAP: {pdp_shap_error[max_depth]:.6f}")
        print("\n")
    
    if args.save:
        results_file = os.path.join("local_disagreements.csv")
        # Make the file if it does not exist
        if not os.path.exists(results_file):
            with open(results_file, 'w') as file:
                file.write("dataset,model,seed,partition,background,max_depth,disagreement\n")
        # Append new results to the file
        with open(results_file, 'a') as file:
            for max_depth in [0, 1, 2, 3]:
                file.write(f"{args.data.name},{args.model_name},{state},")
                file.write(f"{args.partition.type},{int(args.background_size):d},{int(max_depth):d},")
                file.write(f"{pdp_shap_error[max_depth]:.6f}\n")

