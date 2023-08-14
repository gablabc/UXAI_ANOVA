
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_trees, load_FDTree, bar
from utils import rank_diff, normalized_l2norm, Data_Config
from data_utils import INTERACTIONS_MAPPING, SCATTER_SHOW

setup_pyplot_font(25)


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
    A = np.load(os.path.join(model_path, "A_global.npy"))
    pdp = A[..., 1:].mean(axis=1)
    phis = np.load(os.path.join(model_path, "phis_global.npy"))
    background_size = phis.shape[0]
    background = x_train[:args.background_size]

    # Compute disagreement for full background
    local_rank_error = [np.zeros(background_size)]
    local_relative_error = [np.zeros(background_size)]
    for i in range(background_size):
        local_rank_error[0][i] = rank_diff(pdp[i], phis[i])
        local_relative_error[0][i] = normalized_l2norm(pdp[i], phis[i])

    # Compare PDP and Shapley Values
    for i in SCATTER_SHOW[args.data.name]:
        plt.figure()
        plt.scatter(background[:, i], pdp[:, i], c='k', alpha=0.5)
        plt.scatter(background[:, i], phis[:, i], alpha=0.5)
        if args.data.name == "bike":
            if i == 2:
                plt.xticks(np.arange(0, 25, 2))
            else:
                plt.xticks(np.arange(0, 45, 10))
        plt.grid('on', zorder=1)
        plt.xlabel(features.names[i])
        plt.ylabel(f"Attribution of {features.names[i]}")
        filename = f"Attribution_{i}.pdf"
        plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')


    colors = ['blue', 'red', 'green', 'orange', 
              'violet', 'brown', 'cyan', 'olive']
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(args.data.name, args.model_name, max_depth)
        groups, rules = tree.predict(background[:, interactions], latex_rules=True)

        # Store the disagreements here
        local_rank_error.append( np.zeros(background_size) )
        local_relative_error.append( np.zeros(background_size) )

        # Explain in each Region
        phis = [0] * tree.n_groups
        pdps = [0] * tree.n_groups
        backgrounds = [0] * tree.n_groups
        for group_idx in range(tree.n_groups):
            idx_select = (groups == group_idx)
            regional_background = background[idx_select]
            backgrounds[group_idx] = regional_background

            # SHAP
            filename = f"phis_max_depth_{max_depth}_region_{group_idx}.npy"
            phis[group_idx] = np.load(os.path.join(model_path, filename))

            # Reshape idx to index the A matrix
            idx_select = np.where(idx_select)[0].reshape((-1, 1))
            pdps[group_idx] = A[..., 1:][idx_select, idx_select.T].mean(axis=1)

            ########### Local Feature Attribution ############

            # Show scatter plots of PDP and SHAP for features Hour and Temperature

            for i, idx in enumerate(idx_select.ravel()):
                local_rank_error[-1][idx] = rank_diff(pdps[group_idx][i], phis[group_idx][i])
                local_relative_error[-1][idx] = normalized_l2norm(pdps[group_idx][i], phis[group_idx][i])


        # Show scatter plots of PDP and SHAP for features Hour and Temperature
        for i in SCATTER_SHOW[args.data.name]:
            plt.figure()
            for p in range(tree.n_groups):
                plt.scatter(backgrounds[p][:, i], phis[p][:, i], alpha=0.5, 
                            c=colors[p], label=rules[p], zorder=3)
                plt.scatter(backgrounds[p][:, i], pdps[p][:, i], c='k', 
                            alpha=0.5, zorder=3)
            legend_labels = plt.gca().get_legend_handles_labels()
            if args.data.name == "bike":
                if i == 2:
                    plt.xticks(np.arange(0, 25, 2))
                else:
                    plt.xticks(np.arange(0, 45, 10))
            plt.grid('on', zorder=1)
            plt.xlabel(features.names[i])
            plt.ylabel(f"Attribution of {features.names[i]}")
            filename = f"Attribution_{i}_max_depth_{max_depth}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')
        
        # Plot the legend separately
        fig_leg = plt.figure(figsize=(5, 0.6))
        ax_leg = fig_leg.add_subplot(111)
        # Add the legend from the previous axes
        ax_leg.legend(*legend_labels, loc='center', ncol=2, prop={"size": 10})
        # Hide the axes frame and the x/y labels
        ax_leg.axis('off')
        filename = f"Legend_max_depth_{max_depth}.pdf"
        plt.savefig(os.path.join(image_path, filename), bbox_inches='tight', pad_inches=0)

    for max_depth in [0, 1, 2, 3]:
        print(f"Max Depth : {max_depth}")
        rank_jump = int(local_rank_error[max_depth].max())
        print(f"Worst Rank Jump : {rank_jump:d}")
        mean = local_relative_error[max_depth].mean()
        std = local_relative_error[max_depth].std()
        print(f"Local Relative Disagreement : {mean:.2f} +- {std:.2f}")
        print("\n")
