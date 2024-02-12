
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import urllib

# Local imports
from utils import COLORS
from utils import setup_pyplot_font, setup_data_trees, custom_train_test_split
from utils import load_FDTree
from data_utils import INTERACTIONS_MAPPING

setup_pyplot_font(20)


def visualize_tree(tree, xlim, ylim):
    ax = plt.gca()
    idx = [0]

    # Plot the decision boundaries
    def plot_boundaries(node, xlim, ylim):
        # Reached a Leaf
        if node.child_left is None:
            # Plot the compatible leaves
            rect = Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                                facecolor=COLORS[idx[0]], alpha=0.25)
            ax.add_patch(rect)
            idx[0] += 1
        else:
            # Split along x axis
            if node.feature == 3:
                ax.plot([node.threshold, node.threshold], ylim, '-k', zorder=3)
                plot_boundaries(node.child_left, [xlim[0], node.threshold], ylim)
                plot_boundaries(node.child_right, [node.threshold, xlim[1]], ylim)
        
            elif node.feature == 2:
                ax.plot(xlim, [node.threshold, node.threshold], '-k', zorder=3)
                plot_boundaries(node.child_left, xlim, [ylim[0], node.threshold])
                plot_boundaries(node.child_right, xlim, [node.threshold, ylim[1]])
    
    plot_boundaries(tree.root, xlim, ylim)


if __name__ == "__main__":

    # Load data and model
    X, y, features, task = setup_data_trees("california")
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
    interactions = INTERACTIONS_MAPPING["california"]

    # Download the California image
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    filename = "california.png"
    url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
    urllib.request.urlretrieve(url, os.path.join("Images", filename))
    california_img=mpimg.imread(os.path.join("Images", filename))

    # Load the FD-Tree
    tree = load_FDTree(2, "california", "gbt", 2, "pfi", 600)
    tree.print()

    plt.figure()
    visualize_tree(tree, (-124.55, -113.80), (32.45, 42.05))
    plt.scatter(x_train[:1000, -1], x_train[:1000, -2], c='k', s=5)
    # Los Angeles, San Francisco, San Diego, San Jose
    big_cities = \
        np.array([
            (34.052235, -118.243683),
            (37.773972, -122.431297),
            (32.715736, -117.161087),
            (37.352390, -121.953079),
        ])
    plt.scatter(big_cities[:, 1], big_cities[:, 0], c='r', s=100, marker="*", zorder=2)
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    plt.savefig(os.path.join("Images", "splitted_california.pdf"), bbox_inches='tight')
    # plt.show()

