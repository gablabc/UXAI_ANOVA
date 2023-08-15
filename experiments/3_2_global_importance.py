
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_trees, load_FDTree, bar
from utils import rank_diff, normalized_l2norm, Data_Config
from data_utils import INTERACTIONS_MAPPING

setup_pyplot_font(20)

sys.path.append(os.path.abspath(".."))
from src.anova import get_PFI


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_argument("--model_name", type=str, default="rf", 
                       help="Type of tree ensemble either gbt or rf")
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
    phis = np.load(os.path.join(model_path, "phis_global.npy"))
    background_size = phis.shape[0]
    background = x_train[:background_size]

    # Measure of non-additivity
    f = A.sum(-1)[np.arange(background_size), np.arange(background_size)]
    impurity = np.mean((f - A.sum(-1).mean(1))**2)
    print(f"Non-additivity : {impurity:.2f}")

    # Global Feature Importance

    # PDP feature importance
    pdp = A[..., 1:].mean(axis=1)
    I_PDP = np.std(pdp, axis=0)
    bar(I_PDP, features.names)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(image_path, "PDP_Importance.pdf"), bbox_inches='tight')


    # SHAP feature importance
    I_SHAP = np.sqrt((phis**2).mean(axis=0))
    bar(I_SHAP, features.names)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(image_path, "SHAP_Importance.pdf"), bbox_inches='tight')


    # PFI feature importance
    I_PFI = np.sqrt(get_PFI(A))
    bar(I_PFI, features.names)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(image_path, "PFI_Importance.pdf"), bbox_inches='tight')

    # Average error between explainers
    global_rank_error = [[rank_diff(I_PDP, I_SHAP),
                        rank_diff(I_PDP, I_PFI),
                        rank_diff(I_PFI, I_SHAP)]]
    global_relative_error = [[normalized_l2norm(I_PDP, I_SHAP),
                            normalized_l2norm(I_PDP, I_PFI),
                            normalized_l2norm(I_PFI, I_SHAP)]]
    plt.close('all')
    
    for max_depth in [1, 2, 3]:

        # Load the FD-Tree
        tree = load_FDTree(args.data.name, args.model_name, max_depth)
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
            filename = f"phis_max_depth_{max_depth}_region_{group_idx}.npy"
            phis = np.load(os.path.join(model_path, filename))

            # Reshape idx to index the A matrix
            idx_select = np.where(idx_select)[0].reshape((-1, 1))


            ########### Global Feature Importance ############


            # PDP feature importance
            pdp = A[..., 1:][idx_select, idx_select.T].mean(axis=1)
            I_PDP = np.std(pdp, axis=0)
            bar(I_PDP, features.names)
            plt.yticks(fontsize=15)
            filename = f"PDP_Importance_max_depth_{max_depth}_region_{group_idx}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')

            # SHAP feature importance
            I_SHAP = np.sqrt((phis**2).mean(axis=0))
            bar(I_SHAP, features.names)
            plt.yticks(fontsize=15)
            filename = f"SHAP_Importance_max_depth_{max_depth}_region_{group_idx}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')

            # PFI feature importance
            I_PFI = np.sqrt(get_PFI(A[idx_select, idx_select.T]))
            bar(I_PFI, features.names)
            plt.yticks(fontsize=15)
            filename = f"PFI_Importance_max_depth_{max_depth}_region_{group_idx}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')
            plt.close('all')

            # Disagreement betwene global explanations
            global_rank_error[-1][group_idx] = [rank_diff(I_PDP, I_SHAP),
                                                rank_diff(I_PDP, I_PFI),
                                                rank_diff(I_PFI, I_SHAP)]
            global_relative_error[-1][group_idx] = [normalized_l2norm(I_PDP, I_SHAP),
                                                    normalized_l2norm(I_PDP, I_PFI),
                                                    normalized_l2norm(I_PFI, I_SHAP)]
            

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

