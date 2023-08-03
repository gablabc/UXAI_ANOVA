""" Toy Example with Correlations """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap
from shap.maskers import Independent

from utils import get_all_tree_preds, setup_pyplot_font
sys.path.append(os.path.abspath(".."))
from src.anova_tree import ANOVATree
from src.anova import get_ANOVA_1, get_ANOVA_2

setup_pyplot_font(20)

# Generate the data
np.random.seed(42)
rho = 0.75
cov = rho * np.ones((2, 2)) + (1 - rho) * np.eye(2)
X = np.random.multivariate_normal([0, 0], cov, size=1000)
f = lambda X: 2*X[:, 0] + 2*X[:, 1]
y = f(X)
print(np.min(y), np.max(y))
n_estimators = 50
rf = RandomForestRegressor(random_state=2, max_depth=5, 
                           n_estimators=n_estimators, n_jobs=1)
rf.fit(X, y)

# Plot Data + model uncertainty
XX, YY = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
ZZ = get_all_tree_preds(np.column_stack((XX.ravel(), YY.ravel())), rf)
uncertainty = np.log(np.var(ZZ, axis=1).reshape(XX.shape))

plt.figure()
varmap = plt.contourf(XX, YY, uncertainty, cmap='Blues', alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=2)
plt.colorbar(varmap)
plt.savefig(os.path.join("Images", "correlations", f"extrapolation.pdf"), 
                                                    bbox_inches='tight')



############# Vanilla Setup #############


# Run SHAP on whole dataset
background = X
mu = rf.predict(X).mean()
masker = Independent(background, max_samples=background.shape[0])
explainer = shap.explainers.Exact(rf.predict, masker)
phis = explainer(background).values

# ANOVA Additive Decomposition
A = get_ANOVA_1(X, rf.predict)

# Compare PDP and Shapley Values
for i in range(2):
    plt.figure()
    plt.scatter(X[:, i], A[..., i+1].mean(1), c='k', alpha=0.5)
    plt.scatter(X[:, i], phis[:, i], alpha=0.5)
    # argsort_x_i = np.argsort(X[:, i])
    # for n in range(20):
    #     plt.plot(X[argsort_x_i, i], A[:, n, i+1][argsort_x_i], 'k--', alpha=0.5)
    plt.xlabel(r"$\bm{x}_" + str(i) + "$")
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.savefig(os.path.join("Images", "correlations", f"attrib_feature_{i}.pdf"), 
                                                                bbox_inches='tight')



############# Anova-Tree #############

# Fit the tree
tree = ANOVATree(["x0", "x1"], max_depth=2, 
                 save_losses=True, negligible_impurity=0.1)
tree.fit(X, A.sum(-1))
tree.print(verbose=True)
groups = tree.predict(X)


# Plot the objective values w.r.t the split candidates
plt.figure()
for i in range(2):
    splits = tree.root.splits[i]
    objectives = tree.root.objectives[i]
    plt.plot(splits, objectives, '-o', label=f"x{i}")
plt.xlabel(f"Split value")
plt.ylabel(r"$L_2$ Cost of Exclusion")
plt.legend()
plt.savefig(os.path.join("Images", "correlations", "L2_Exclusion.pdf"), 
                                                        bbox_inches='tight')


# ANOVA Order-2 Decomposition
F = get_ANOVA_2(X, rf.predict, [0, 1])
idx = np.argmax(X[:, 0])
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=A[:, idx, 2]**2, cmap='Blues', 
                                    edgecolor='k', alpha=0.75)
plt.text(X[idx, 0], X[idx, 1]+0.1, r"$\bm{z}$", fontsize=30, horizontalalignment="center")
plt.colorbar()
plt.savefig(os.path.join("Images", "correlations", "Interactions.pdf"), 
                                                        bbox_inches='tight')



# Rerun SHAP and PDP
phis = [0] * tree.n_groups
pdps = [0] * tree.n_groups
backgrounds = [0] * tree.n_groups
colors = ['blue', 'red', 'green', 'orange']
for group_idx in range(tree.n_groups):
    idx_select = (groups == group_idx)
    background = X[idx_select]
    backgrounds[group_idx] = background

    # SHAP
    mu = rf.predict(background).mean()
    masker = Independent(background, max_samples=background.shape[0])
    explainer = shap.explainers.Exact(rf.predict, masker)
    phis[group_idx] = explainer(background).values

    # PDP
    idx_select = np.where(idx_select)[0].reshape((-1, 1))
    pdps[group_idx] = A[..., 1:][idx_select, idx_select.T].mean(1)



# Compare SHAP and PDP on each leaf
for i in range(2):
    plt.figure()
    for p in range(tree.n_groups):
        plt.scatter(backgrounds[p][:, i], pdps[p][:, i], alpha=0.5, c='k')
        plt.scatter(backgrounds[p][:, i], phis[p][:, i], alpha=0.5, 
                    c=colors[p], label=f"Group {p}")
    plt.xlabel(r"$\bm{x}_" + str(i) + "$")
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.legend()
    plt.savefig(os.path.join("Images", "correlations", 
                    f"attrib_feature_{i}_regions.pdf"), bbox_inches='tight')


