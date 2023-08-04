""" Second example in the paper """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap.maskers import Independent
from sklearn.neural_network import MLPRegressor

# Local imports
from utils import setup_pyplot_font
sys.path.append(os.path.abspath(".."))
from src.features import Features
from src.anova_tree import FDTree
from src.anova import get_ANOVA_1

setup_pyplot_font(20)

# Generate the data
np.random.seed(42)
d = 4
feature_names_latex = [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$"]
X = np.random.uniform(-1, 1, size=(1000, d))
features = Features(X, [f"x{i}" for i in range(d)], ["num"]*d)
def h(X):
    y_hat = np.zeros((X.shape[0]))
    mask = (X[:, 0] > 0) & (X[:, 1] > 0)
    y_hat[mask] = np.sin(np.pi * X[mask, 2])
    y_hat[~mask] = -2 * X[~mask, 2]**2 + X[~mask, 3]
    return y_hat
y = h(X)
model = MLPRegressor(hidden_layer_sizes=(100, 50, 20, 10), max_iter=500).fit(X, y)
h = lambda x : model.predict(x)


############# Vanilla Setup #############


# Run SHAP on whole dataset
background = X
mu = h(background).mean()
masker = Independent(background, max_samples=background.shape[0])
explainer = shap.explainers.Exact(h, masker)
phis = explainer(background).values

# ANOVA Additive Decomposition
A = get_ANOVA_1(background, h)

# Compare PDP and Shapley Values
for i in range(4):
    plt.figure()
    plt.scatter(background[:, i], A[..., i+1].mean(1), c='k', alpha=0.5)
    plt.scatter(background[:, i], phis[:, i], alpha=0.5)
    plt.xlabel(feature_names_latex[i])
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.ylim(-1, 1)


############# Anova-Tree #############

# Fit the tree
tree = FDTree(features, max_depth=2, save_losses=True)
tree.fit(X, A.sum(-1))
tree.print(verbose=True)
groups, rules = tree.predict(X)
print(rules)

# Plot the objective values w.r.t the split candidates
plt.figure()
for i in range(3):
    splits = tree.root.splits[i]
    objectives = tree.root.objectives[i]
    plt.plot(splits, objectives, '-o', label=feature_names_latex[i])
plt.ylim(0, y.var())
plt.xlabel(f"Split value")
plt.ylabel(r"$L_2$ Cost of Exclusion")
plt.legend()



# Rerun SHAP and PDP
print("Reruning SHAP")
pdps = [0] * tree.n_groups
phis = [0] * tree.n_groups
backgrounds = [0] * tree.n_groups
colors = ['blue', 'red', 'green', 'orange']
for group_idx in range(tree.n_groups):
    idx_select = (groups == group_idx)
    background = X[idx_select]
    backgrounds[group_idx] = background

    # PDP
    idx_select = np.where(idx_select)[0].reshape((-1, 1))
    pdps[group_idx] = A[..., 1:][idx_select, idx_select.T].mean(1)

    # SHAP
    mu = h(background).mean()
    masker = Independent(background, max_samples=background.shape[0])
    explainer = shap.explainers.Exact(h, masker)
    phis[group_idx] = explainer(background).values


# Compare SHAP and PDP on each leaf
for i in range(4):
    plt.figure()
    for p in range(tree.n_groups):
        plt.scatter(backgrounds[p][:, i], pdps[p][:, i], c='k', alpha=0.5)
        plt.scatter(backgrounds[p][:, i], phis[p][:, i], alpha=0.5, 
                    c=colors[p], label=f"Group {p}")
    plt.xlabel(feature_names_latex[i])
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.ylim(-1, 1)
    plt.legend()
plt.show()