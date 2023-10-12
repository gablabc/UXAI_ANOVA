""" Toy Example of how GADGET-PDP works """

import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local
from utils import setup_pyplot_font

sys.path.append(os.path.abspath(".."))
from src.anova import get_ANOVA_1

setup_pyplot_font(30)

image_path = os.path.join("Images", "GADGET")

# Generate the data
np.random.seed(42)
d = 2
X = np.random.uniform(-1, 1, size=(1000, 2))

# Model with weak interactions
def h_add(X):
    return 2.5 * X[:, 0] - 2 * np.sin(2*np.pi*X[:, 1]) + 0.75 * X[:, 0] * X[:, 1]

# Model with interactions
def h_inter(X):
    return 2 * np.sin(2*np.pi*(X[:, 0] + X[:, 1])) + 1.5 * X[:, 0]


############# Additive #############

# ANOVA Additive Decomposition
A = get_ANOVA_1(X, h_add)
h_preds = A[0, :, 0]
A = A + h_preds.reshape((1, -1, 1))
A = A[..., 1:]  # (N, N, d) tensor s.t. A_ijk = h_(r_{k}(x^(j), x^(i)))
A_c = A - A.mean(0, keepdims=True)

for k in range(2):
    argsort_x_i = np.argsort(X[:, k])
    # Show the uncentered ICE curves
    plt.figure()
    for n in range(6):
        plt.plot(X[argsort_x_i, k], A[:, n, k][argsort_x_i], alpha=0.5, linewidth=4)
    plt.ylim(-4, 4)
    plt.xlim(-1, 1)
    plt.xlabel(r"$x_k$")
    plt.savefig(os.path.join(image_path, f"ICE_add_feature_{k}.pdf"), bbox_inches='tight')

    # Show the centered ICE curves
    plt.figure()
    for n in range(6):
        plt.plot(X[argsort_x_i, k], A_c[:, n, k][argsort_x_i], alpha=0.5, linewidth=4)
    # Show the mean centered PDP
    plt.plot(X[argsort_x_i, k], np.mean(A_c[..., k], 1)[argsort_x_i], "k--", linewidth=6)
    plt.ylim(-4, 4)
    plt.xlim(-1, 1)
    plt.xlabel(r"$x_k$")
    plt.savefig(os.path.join(image_path, f"ICE_centered_add_feature_{k}.pdf"), bbox_inches='tight')



############# Non-Additive #############

# ANOVA Additive Decomposition
A = get_ANOVA_1(X, h_inter)
h_preds = A[0, :, 0]
A = A + h_preds.reshape((1, -1, 1))
A = A[..., 1:]  # (N, N, d) tensor s.t. A_ijk = h_(r_{k}(x^(j), x^(i)))
A_c = A - A.mean(0, keepdims=True)

for k in range(2):
    argsort_x_i = np.argsort(X[:, k])
    # Show the uncentered ICE curves
    plt.figure()
    for n in range(6):
        plt.plot(X[argsort_x_i, k], A[:, n, k][argsort_x_i], alpha=0.5, linewidth=4)
    plt.ylim(-4, 4)
    plt.xlim(-1, 1)
    plt.xlabel(r"$x_k$")
    plt.savefig(os.path.join(image_path, f"ICE_inter_feature_{k}.pdf"), bbox_inches='tight')


    # Show the centered ICE curves
    plt.figure()
    for n in range(6):
        plt.plot(X[argsort_x_i, k], A_c[:, n, k][argsort_x_i], alpha=0.5, linewidth=4)
    # Show the mean centered PDP
    plt.plot(X[argsort_x_i, k], np.mean(A_c[..., k], 1)[argsort_x_i], "k--", linewidth=6)
    plt.ylim(-4, 4)
    plt.xlim(-1, 1)
    plt.xlabel(r"$x_k$")
    plt.savefig(os.path.join(image_path, f"ICE_centered_inter_feature_{k}.pdf"), bbox_inches='tight')
