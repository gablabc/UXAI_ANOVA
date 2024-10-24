""" Toy Example with Correlations """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from utils import get_all_tree_preds, setup_pyplot_font, plot_legend, COLORS
sys.path.append(os.path.abspath(".."))
from src.features import Features
from src.anova_tree import PARTITION_CLASSES
from src.anova import get_ANOVA_1, get_ANOVA_2, interventional_treeshap

setup_pyplot_font(20)
image_path = os.path.join("Images", "correlations")

# Generate the data
np.random.seed(42)
rho = 0.75
cov = rho * np.ones((2, 2)) + (1 - rho) * np.eye(2)
X = np.random.multivariate_normal([0, 0], cov, size=1000)
features = Features(X, [r"$x_0$", r"$x_1$"], ["num", "num"])
f = lambda X: 2*X[:, 0] + 2*X[:, 1]
y = f(X)
rf = RandomForestRegressor(random_state=2, max_depth=5, 
                           n_estimators=50, n_jobs=1)
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
plt.savefig(os.path.join(image_path, f"extrapolation.pdf"), bbox_inches='tight')



############# Vanilla Setup #############


# Run SHAP on whole dataset
background = X
mu = rf.predict(X).mean()
phis, _ = interventional_treeshap(rf, background, background)

# ANOVA Additive Decomposition
H = get_ANOVA_1(X, rf.predict)

# Compare PDP and Shapley Values
for i in range(2):
    plt.figure()
    sorted_idx = np.argsort(background[:, i])
    plt.plot(background[sorted_idx, i], H[..., i+1].mean(1)[sorted_idx], 'k-')
    plt.scatter(X[:200, i], phis[:200, i], alpha=0.25, c='k')
    # argsort_x_i = np.argsort(X[:, i])
    # for n in range(20):
    #     plt.plot(X[argsort_x_i, i], A[:, n, i+1][argsort_x_i], 'k--', alpha=0.5)
    plt.xlabel(r"$\bm{x}_" + str(i) + "$")
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.savefig(os.path.join(image_path, f"attrib_feature_{i}.pdf"), bbox_inches='tight')



# ANOVA Order-2 Decomposition
F = get_ANOVA_2(X, rf.predict, [0, 1])
idx = np.argmax(X[:, 0])
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=H[:, idx, 2]**2, cmap='Blues', 
                                    edgecolor='k', alpha=0.75)
plt.text(X[idx, 0], X[idx, 1]+0.1, r"$\bm{z}$", fontsize=30, horizontalalignment="center")
plt.colorbar()
plt.savefig(os.path.join(image_path, "Interactions.pdf"), bbox_inches='tight')



############# Anova-Tree #############

# Fit the tree
partition_type = "l2coe"
# partition_type = "pfi"
# partition_type = "gadget-pdp"
tree = PARTITION_CLASSES[partition_type](features, max_depth=2, save_losses=True, negligible_impurity=1)
if partition_type == "l2coe":
    tree.fit(X, H.sum(-1))
elif partition_type == "pfi":
    tree.fit(X, H)
elif partition_type == "gadget-pdp":
    tree.fit(X, H)

# Print results
tree.print(verbose=True)
groups, rules = tree.predict(X, latex_rules=True)
print(rules)


# Plot the objective values w.r.t the split candidates
plt.figure()
for i in range(2):
    splits = tree.root.splits[i]
    objectives = tree.root.objectives[i]
    plt.plot(splits, objectives, '-o', label=f"x{i}")
plt.xlabel(f"Split value")
if partition_type == "l2coe":
    plt.ylabel(r"$L_2$ Cost of Exclusion")
elif partition_type == "pfi":
    plt.ylabel("Loss PFI vs PDP")
elif partition_type == "gadget-pdp":
    plt.ylabel("Loss ICE vs PDP")
plt.legend()
plt.savefig(os.path.join(image_path, "L2_Exclusion.pdf"), bbox_inches='tight')


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
    phis[group_idx], _ = interventional_treeshap(rf, background, background)

    # PDP
    idx_select = np.where(idx_select)[0].reshape((-1, 1))
    pdps[group_idx] = H[..., 1:][idx_select, idx_select.T].mean(1)



# Compare SHAP and PDP on each leaf
for i in range(2):
    plt.figure()
    for p in range(tree.n_groups):
        sorted_idx = np.argsort(backgrounds[p][:, i])
        plt.plot(backgrounds[p][sorted_idx, i], pdps[p][sorted_idx, i], 'k-')
        plt.scatter(backgrounds[p][:200, i], phis[p][:200, i], alpha=0.25, c=COLORS[p])
    plt.xlabel(r"$\bm{x}_" + str(i) + "$")
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.savefig(os.path.join(image_path, f"attrib_feature_{i}_regions.pdf"), bbox_inches='tight')

plot_legend(rules, ncol=3)
plt.savefig(os.path.join(image_path, "Legend.pdf"), bbox_inches='tight', pad_inches=0)
