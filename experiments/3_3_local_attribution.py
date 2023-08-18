
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import COLORS, plot_legend, attrib_scatter_plot
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_trees, load_FDTree
from utils import pdp_vs_shap, Data_Config
from data_utils import INTERACTIONS_MAPPING, SCATTER_SHOW

setup_pyplot_font(25)


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
    parser.add_argument("--ncol", type=int, default=2, 
                       help="Number of columns in the Legend")
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
    background = x_train[:background_size]

    # Compute disagreement for full background
    pdp_shap_error = [pdp_vs_shap(pdp, phis)]

    # Compare PDP and Shapley Values
    for i in SCATTER_SHOW[args.data.name]:
        # For bike we have specific xticks
        attrib_scatter_plot(background, pdp, phis, i, features, args)
        filename = f"Attribution_{i}.pdf"
        plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')


    # For various depths of FD-Tree
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(args.data.name, args.model_name, max_depth)
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
            filename = f"phis_max_depth_{max_depth}_region_{group_idx}.npy"
            phis[group_idx] = np.load(os.path.join(model_path, filename))

            # Reshape idx to index the A matrix
            idx_select = np.where(idx_select)[0].reshape((-1, 1))
            pdps[group_idx] = A[..., 1:][idx_select, idx_select.T].mean(axis=1)

            ########### Local Feature Attribution ############

            # Compute disagreement
            pdp_shap_error[-1] += pdp_vs_shap(pdps[group_idx], phis[group_idx]) * len(idx_select)    
        pdp_shap_error[-1] /= background_size


        # Show scatter plots of PDP and SHAP for features Hour and Temperature
        for i in SCATTER_SHOW[args.data.name]:
            plt.figure()
            attrib_scatter_plot(backgrounds, pdps, phis, i, features, args)
            filename = f"Attribution_{i}_max_depth_{max_depth}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')
        
        # Plot the legend separately
        plot_legend(rules, ncol=args.ncol)
        filename = f"Legend_max_depth_{max_depth}.pdf"
        plt.savefig(os.path.join(image_path, filename), bbox_inches='tight', pad_inches=0)

    for max_depth in [0, 1, 2, 3]:
        print(f"PDP vs SHAP: {pdp_shap_error[max_depth]}")
        print("\n")
