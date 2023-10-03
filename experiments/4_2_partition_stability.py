
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import itertools

# Local imports
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import get_background
from data_utils import INTERACTIONS_MAPPING

from src.anova_tree import Partition, PARTITION_CLASSES

setup_pyplot_font(25)


if __name__ == "__main__":
    
    image_path = os.path.join("Images", "Stability")
    # Make folder for dataset models
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Load data and model
    data_names = ["bike", "adult_income", "marketing", "default_credit", "kin8nm", "california"]
    model_names = ["gbt", "rf"]
    for data in data_names:
        X, y, features, task = setup_data_trees(data)
        x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
        interactions = INTERACTIONS_MAPPING[data]
        subset_features = features.select(interactions)

        for model_name in model_names:

            print(f"Data : {data}   Model : {model_name}")
            # Prepare the arrays to fill
            N_sample_candidates = list(range(100, 801, 25))
            partitions = np.zeros((10, X.shape[0], len(N_sample_candidates), 3))
            agreements = np.zeros((len(N_sample_candidates), 3))

            # Folder for dataset models
            model_path = os.path.join("models", data, model_name + "_0")

            # Get the pre-computed A matrix
            A = np.load(os.path.join(model_path, f"A_global_N_1000.npy"))
            background = get_background(x_train, 1000, 0)

            # Iterate over all subsample random seeds
            for state in range(10):
                # Grow the size of the backgrouns
                for j, N_samples in enumerate(N_sample_candidates):
                    # Subsample the background and A
                    np.random.seed(state)
                    select = np.random.choice(range(1000), N_samples, replace=False)
                    background_ = background[select]
                    select = select.reshape((-1, 1))
                    A_ = A[select, select.T]
                    A_ = A_.sum(-1)
                    
                    # For all max depths
                    for k, max_depth in enumerate([1, 2, 3]):

                        # Use the partitioning tree
                        Tree = PARTITION_CLASSES["l2coe"]
                        
                        # Full growth
                        tree = Tree(subset_features, max_depth=max_depth,
                                    save_losses=False,
                                    negligible_impurity=0,
                                    relative_decrease=2,
                                    samples_leaf=10)
                        tree.fit(background_[:, interactions], A_)
                        # tree.print()
                        groups, rules = tree.predict(X[:, interactions])

                        # Make sure the tree was fully grown
                        if not len(np.unique(groups)) == 2**max_depth:
                            tree.fit(background_[:, interactions], A_)
                    
                        partitions[state, :, j, k] = groups


            # Compute the partition stability
            for j, N_samples in enumerate(N_sample_candidates):
                for k, max_depth in enumerate([1, 2, 3]):

                    # Compute agreements between each pair of seeds
                    for pair in itertools.combinations(range(10), 2):
                        agreements[j, k] += adjusted_rand_score(partitions[pair[0], :, j, k],
                                                                partitions[pair[1], :, j, k])
            agreements /= 45


            # Plot the results
            plt.figure()
            plt.plot(N_sample_candidates, agreements, '-o')
            plt.xlabel("Subsample size")
            plt.ylabel("Adjusted Rand Index")
            plt.ylim(0, 1)
            filename = f"Stability_data_{data}_model_{model_name}.pdf"
            plt.savefig(os.path.join(image_path, filename), bbox_inches='tight')