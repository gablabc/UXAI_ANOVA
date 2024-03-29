
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import plot_legend, attrib_scatter_plot
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_FDTree, get_background
from utils import correlation, rank_correlation, l2_disagreement, l2_norm, Data_Config, TreeEnsembleHP
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
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--background_size", type=int, default=600,
                       help="Size of the background data")
    parser.add_argument("--ncol", type=int, default=2, 
                       help="Number of columns in the Legend")
    parser.add_argument("--save", action='store_true', help="Save disagreement metrics")
    parser.add_argument("--plot", action='store_true', help="Plots the local feature attributions")
    args, unknown = parser.parse_known_args()
    print(args)
    
    args.plot = True
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
    H = np.load(os.path.join(model_path, f"A_global_N_{args.background_size}.npy"))
    pdp = H[..., 1:].mean(axis=1)
    phis = np.load(os.path.join(model_path, f"phis_global_N_{args.background_size}.npy"))
    background = get_background(x_train, args.background_size, args.ensemble.random_state)

    # Compare PDP and Shapley Values
    if args.plot:
        for i in SCATTER_SHOW[args.data.name]:
            # For bike we have specific xticks
            attrib_scatter_plot(background, pdp, phis, i, features, args)
            filename = f"Attribution_{i}_N_{args.background_size}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')

    # Load the l2coe to get the model variance
    tree = load_FDTree(1, args.data.name, args.model_name, args.ensemble.random_state, 
                        "l2coe", args.background_size)
    disagreement_factor = tree.impurity_factor

        # Compute disagreement for full background
    pdp_shap_l2_disagreement = [disagreement_factor * l2_disagreement(pdp, phis)]
    pdp_shap_l2_norm = [disagreement_factor * l2_norm(pdp, phis)]
    pdp_shap_pearson_disagreement = [correlation(pdp, phis)]
    pdp_shap_spearman_disagreement = [rank_correlation(np.abs(pdp), np.abs(phis))]

    # For various depths of FD-Tree
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(max_depth, args.data.name, args.model_name, args.ensemble.random_state, 
                           args.partition.type, args.background_size)
        groups, rules = tree.predict(background[:, interactions], latex_rules=True)
        # disagreement_factor = tree.impurity_factor
        # print(tree.impurity_factor)

        # Store the disagreements here
        pdp_shap_l2_disagreement.append(0)
        pdp_shap_l2_norm.append(0)
        pdp_shap_pearson_disagreement.append(0)
        pdp_shap_spearman_disagreement.append(0)

        # Explain in each Region
        phis = [0] * tree.n_groups
        pdps = [0] * tree.n_groups
        backgrounds = [0] * tree.n_groups
        for group_idx in range(tree.n_groups):
            idx_select = (groups == group_idx)
            regional_background = background[idx_select]
            backgrounds[group_idx] = regional_background
            ratio = sum(idx_select) / args.background_size

            # SHAP
            filename = f"phis_{args.partition.type}_N_{args.background_size}_"+\
                       f"max_depth_{max_depth}_region_{group_idx}.npy"
            phis[group_idx] = np.load(os.path.join(model_path, filename))

            # Reshape idx to index the A matrix
            idx_select = np.where(idx_select)[0].reshape((-1, 1))
            pdps[group_idx] = H[..., 1:][idx_select, idx_select.T].mean(axis=1)

            ########### Local Feature Attribution ############

            # Compute disagreement
            pdp_shap_l2_disagreement[-1] += l2_disagreement(pdps[group_idx], phis[group_idx]) * ratio * disagreement_factor
            pdp_shap_l2_norm[-1] += l2_norm(pdps[group_idx], phis[group_idx]) * ratio * disagreement_factor
            pdp_shap_pearson_disagreement[-1] += correlation(pdps[group_idx], phis[group_idx]) * ratio
            pdp_shap_spearman_disagreement[-1] += rank_correlation(np.abs(pdps[group_idx]), np.abs(phis[group_idx])) * ratio

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
    
    if args.save:
        results_file = os.path.join(f"local_disagreements_{state}.csv")
        # Make the file if it does not exist
        if not os.path.exists(results_file):
            with open(results_file, 'w') as file:
                header = "dataset,model,partition,background,max_depth,"
                header += "l2disagreement,l2norm,pearson,spearman\n"
                file.write(header)
        # Append new results to the file
        with open(results_file, 'a') as file:
            for max_depth in [0, 1, 2, 3]:
                file.write(f"{args.data.name},{args.model_name},")
                file.write(f"{args.partition.type},{int(args.background_size):d},{int(max_depth):d},")
                file.write(f"{pdp_shap_l2_disagreement[max_depth]:.8f},")
                file.write(f"{pdp_shap_l2_norm[max_depth]:.8f},")
                file.write(f"{pdp_shap_pearson_disagreement[max_depth]:.8f},")
                file.write(f"{pdp_shap_spearman_disagreement[max_depth]:.8f}\n")

