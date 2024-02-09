
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import COLORS
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_FDTree, three_bars, get_background
from utils import correlation, rank_correlation, l2_norm, l2_disagreement
from utils import Data_Config, TreeEnsembleHP
from data_utils import INTERACTIONS_MAPPING

setup_pyplot_font(20)

sys.path.append(os.path.abspath(".."))
from src.anova import get_PFI
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
    parser.add_argument("--save", action='store_true', help="Save disagreement metrics")
    parser.add_argument("--plot", action='store_true', help="Plots the global feature importance")
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
    phis = np.load(os.path.join(model_path, f"phis_global_N_{args.background_size}.npy"))
    # Background data
    background = get_background(x_train, args.background_size, args.ensemble.random_state)

    # Measure of non-additivity
    f = A.sum(-1)[np.arange(args.background_size), np.arange(args.background_size)]
    impurity = np.mean((f - A.sum(-1).mean(1))**2)
    print(f"Non-additivity : {impurity:.2f}")

    # Global Feature Importance
    pdp = A[..., 1:].mean(axis=1)
    I_PDP = np.std(pdp, axis=0)
    I_SHAP = np.sqrt((phis**2).mean(axis=0))
    I_PFI = np.sqrt(get_PFI(A))

    # Bar chart
    if args.plot:
        three_bars(I_PFI, I_SHAP, I_PDP, features, sort=True)
        plt.yticks(fontsize=15)
        plt.xlabel("Feature Importance")
        filename = f"Importance_N_{args.background_size}.pdf"
        plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')

    # Load the fp-tree to get the model variance
    tree = load_FDTree(1, args.data.name, args.model_name, args.ensemble.random_state, 
                        "l2coe", args.background_size)
    disagreement_factor = tree.impurity_factor

    # Average error between explainers
    global_l2_disagreement = np.zeros(4)
    global_l2_disagreement[0] = disagreement_factor * l2_disagreement(I_PDP, I_SHAP, I_PFI)
    global_l2_norm = np.zeros(4)
    global_l2_norm[0] = disagreement_factor * l2_norm(I_PDP, I_SHAP, I_PFI)
    global_pearson_disagreement = np.zeros(4)
    global_pearson_disagreement[0] = correlation(I_PDP, I_SHAP, I_PFI)
    global_spearman_disagreement = np.zeros(4)
    global_spearman_disagreement[0] = rank_correlation(I_PDP, I_SHAP, I_PFI)
    
    # For various depths of FD-Tree
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(max_depth, args.data.name, args.model_name, args.ensemble.random_state, 
                           args.partition.type, args.background_size)
        groups, rules = tree.predict(background[:, interactions])

        # Store the disagreements here
        l2_disagrement_per_region = np.zeros(tree.n_groups)
        l2_norm_per_region = np.zeros(tree.n_groups)
        pearson_disagrement_per_region = np.zeros(tree.n_groups)
        spearman_disagrement_per_region = np.zeros(tree.n_groups)

        # weight each group by its number of datapoints
        weights = np.zeros(tree.n_groups)

        # Explain in each Region
        backgrounds = [0] * tree.n_groups
        for group_idx in range(tree.n_groups):
            idx_select = (groups == group_idx)
            regional_background = background[idx_select]
            weights[group_idx] = sum(idx_select) / args.background_size

            # SHAP
            filename = f"phis_{args.partition.type}_N_{args.background_size}_" +\
                       f"max_depth_{max_depth}_region_{group_idx}.npy"
            phis = np.load(os.path.join(model_path, filename))

            # Reshape idx to index the A matrix
            idx_select = np.where(idx_select)[0].reshape((-1, 1))


            ########### Global Feature Importance ############


            # PDP feature importance
            pdp = A[..., 1:][idx_select, idx_select.T].mean(axis=1)
            I_PDP = np.std(pdp, axis=0)
            I_SHAP = np.sqrt((phis**2).mean(axis=0))
            I_PFI = np.sqrt(get_PFI(A[idx_select, idx_select.T]))
            
            if args.plot:
                three_bars(I_PFI, I_SHAP, I_PDP, features, color=COLORS[group_idx], sort=True)
                plt.yticks(fontsize=15)
                plt.xlabel("Feature Importance")
                filename = f"Importance_{args.partition.type}_N_{args.background_size}_" +\
                        f"max_depth_{max_depth}_region_{group_idx}.pdf"
                plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')
                plt.close('all')

            # Disagreement betwene global explanations
            l2_disagrement_per_region[group_idx] = l2_disagreement(I_PDP, I_SHAP, I_PFI)
            l2_norm_per_region[group_idx] = l2_norm(I_PDP, I_SHAP, I_PFI)
            pearson_disagrement_per_region[group_idx] = correlation(I_PDP, I_SHAP, I_PFI)
            spearman_disagrement_per_region[group_idx] = rank_correlation(I_PDP, I_SHAP, I_PFI)
        
        assert np.isclose(1, weights.sum())
        global_l2_disagreement[max_depth] = disagreement_factor * np.average(l2_disagrement_per_region, weights=weights)
        global_l2_norm[max_depth] = disagreement_factor * np.average(l2_norm_per_region, weights=weights)
        global_pearson_disagreement[max_depth] = np.average(pearson_disagrement_per_region, weights=weights)
        global_spearman_disagreement[max_depth] = np.average(pearson_disagrement_per_region, weights=weights)


    if args.save:
        results_file = os.path.join(f"global_disagreements_{state}.csv")
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
                file.write(f"{global_l2_disagreement[max_depth]:.8f},")
                file.write(f"{global_l2_norm[max_depth]:.8f},")
                file.write(f"{global_pearson_disagreement[max_depth]:.8f},")
                file.write(f"{global_spearman_disagreement[max_depth]:.8f}\n")