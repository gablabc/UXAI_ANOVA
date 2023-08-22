
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import COLORS
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_trees, load_FDTree, three_bars
from utils import rank_diff, pdp_vs_shap, Data_Config
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
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--background_size", type=int, default=500, 
                       help="Size of the background data")
    parser.add_argument("--save", action='store_true', help="Save disagreement metrics")
    args, unknown = parser.parse_known_args()
    print(args)
    
    # Make folder for dataset models
    folder_path = os.path.join("Images", args.data.name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    image_path = os.path.join(folder_path, args.model_name)
    # Make folder for architecture
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Load data and model
    X, y, features, task = setup_data_trees(args.data.name)
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
    # Load models
    models, perfs = load_trees(args.data.name, args.model_name)
    interactions = INTERACTIONS_MAPPING[args.data.name]

    # Make folder for dataset models
    model_path = os.path.join("models", args.data.name, args.model_name)

    # Get the pre-computed feature attributions
    A = np.load(os.path.join(model_path, f"A_global_N_{args.background_size}.npy"))
    phis = np.load(os.path.join(model_path, f"phis_global_N_{args.background_size}.npy"))
    background = x_train[:args.background_size]

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
    three_bars(I_PFI, I_SHAP, I_PDP, features, sort=True)
    plt.yticks(fontsize=15)
    filename = f"Importance_N_{args.background_size}.pdf"
    plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')

    # Average error between explainers
    global_rank_error = [[rank_diff(I_PDP, I_SHAP),
                        rank_diff(I_PDP, I_PFI),
                        rank_diff(I_PFI, I_SHAP)]]
    global_relative_error = [[pdp_vs_shap(I_PDP, I_SHAP),
                            pdp_vs_shap(I_PDP, I_PFI),
                            pdp_vs_shap(I_PFI, I_SHAP)]]
    plt.close('all')
    
    # For various depths of FD-Tree
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(args.data.name, args.model_name, max_depth,
                           args.partition.type, args.background_size)
        groups, rules = tree.predict(background[:, interactions])

        # Store the disagreements here
        global_rank_error.append( np.zeros((tree.n_groups, 3)) )
        global_relative_error.append( np.zeros((tree.n_groups, 3)) )

        # Explain in each Region
        backgrounds = [0] * tree.n_groups
        for group_idx in range(tree.n_groups):
            idx_select = (groups == group_idx)
            regional_background = background[idx_select]

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
            three_bars(I_PFI, I_SHAP, I_PDP, features, color=COLORS[group_idx], sort=True)
            plt.yticks(fontsize=15)
            filename = f"Importance_{args.partition.type}_N_{args.background_size}_" +\
                       f"max_depth_{max_depth}_region_{group_idx}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')
            plt.close('all')

            # Disagreement betwene global explanations
            global_rank_error[-1][group_idx] = [rank_diff(I_PDP, I_SHAP),
                                                rank_diff(I_PDP, I_PFI),
                                                rank_diff(I_PFI, I_SHAP)]
            global_relative_error[-1][group_idx] = [pdp_vs_shap(I_PDP, I_SHAP),
                                                    pdp_vs_shap(I_PDP, I_PFI),
                                                    pdp_vs_shap(I_PFI, I_SHAP)]
            

    global_rank_error[0] = np.array(global_rank_error[0])
    global_relative_error[0] = np.array(global_relative_error[0])
    for max_depth in [0, 1, 2, 3]:
        print(f"Max Depth : {max_depth}")
        rank_jump = int(global_rank_error[max_depth].max())
        print(f"Worst Rank Jump : {rank_jump:d}")
        mean = global_relative_error[max_depth].mean()
        std = global_relative_error[max_depth].std()
        print(f"Global Relative Disagreement : {mean:.2f} +- {std:.2f}")
        print("\n")

    if args.save:
        results_file = os.path.join("global_disagreements.csv")
        # Make the file if it does not exist
        if not os.path.exists(results_file):
            with open(results_file, 'w') as file:
                file.write("dataset,model,partition,background,max_depth,disagreement\n")
        # Append new results to the file
        with open(results_file, 'a') as file:
            for max_depth in [0, 1, 2, 3]:
                error = global_relative_error[max_depth].mean()
                file.write(f"{args.data.name},{args.model_name},")
                file.write(f"{args.partition.type},{int(args.background_size):d},{int(max_depth):d},")
                file.write(f"{error:.6f}\n")