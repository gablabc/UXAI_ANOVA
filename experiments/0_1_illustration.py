""" Second example in the paper """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap.maskers import Independent
from sklearn.neural_network import MLPRegressor

# Local imports
from utils import setup_pyplot_font, plot_legend, COLORS
sys.path.append(os.path.abspath(".."))
from src.features import Features
from src.anova_tree import PARTITION_CLASSES
from src.anova import get_ANOVA_1

image_path = os.path.join("Images", "Illustration")
setup_pyplot_font(30)

# Generate the data
np.random.seed(42)
d = 4
feature_names_latex = [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$"]
X = np.random.uniform(-1, 1, size=(1000, d))
features = Features(X, feature_names_latex, ["num"]*d)
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
H = get_ANOVA_1(background, h)

# Compare PDP and Shapley Values
for i in range(4):
    plt.figure()
    sorted_idx = np.argsort(background[:, i])
    plt.plot(background[sorted_idx, i], H[..., i+1].mean(1)[sorted_idx], 'k-', linewidth=5)
    plt.scatter(background[:700, i], phis[:700, i], alpha=0.75, c='gray')
    plt.xlabel(feature_names_latex[i])
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.ylim(-1, 1)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.savefig(os.path.join(image_path, f"Attrib_{i}.pdf"), bbox_inches='tight')


############# Anova-Tree #############

# Fit the tree
partition_type = "l2coe"
# partition_type = "pfi"
# partition_type = "gadget-pdp"
tree = PARTITION_CLASSES[partition_type](features, max_depth=2, save_losses=True)
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
for i in range(3):
    splits = tree.root.splits[i]
    objectives = tree.root.objectives[i]
    plt.plot(splits, objectives, '-o', label=feature_names_latex[i])
plt.ylim(0, y.var())
plt.xlabel(f"Split value")
if partition_type == "l2coe":
    plt.ylabel(r"$L_2$ Cost of Exclusion")
elif partition_type == "pfi":
    plt.ylabel("Loss PFI vs PDP")
elif partition_type == "gadget-pdp":
    plt.ylabel("Loss ICE vs PDP")
plt.legend()
plt.savefig(os.path.join(image_path, f"Loss.pdf"), bbox_inches='tight')



# Rerun SHAP and PDP
print("Reruning SHAP")
pdps = [0] * tree.n_groups
phis = [0] * tree.n_groups
backgrounds = [0] * tree.n_groups
for group_idx in range(tree.n_groups):
    idx_select = (groups == group_idx)
    background = X[idx_select]
    backgrounds[group_idx] = background

    # PDP
    idx_select = np.where(idx_select)[0].reshape((-1, 1))
    pdps[group_idx] = H[..., 1:][idx_select, idx_select.T].mean(1)

    # SHAP
    mu = h(background).mean()
    masker = Independent(background, max_samples=background.shape[0])
    explainer = shap.explainers.Exact(h, masker)
    phis[group_idx] = explainer(background).values


# Compare SHAP and PDP on each leaf
for i in range(4):
    plt.figure()
    for p in range(tree.n_groups):
        sorted_idx = np.argsort(backgrounds[p][:, i])
        plt.plot(backgrounds[p][sorted_idx, i], pdps[p][sorted_idx, i], 'k-', linewidth=5)
        plt.plot(backgrounds[p][sorted_idx, i], pdps[p][sorted_idx, i], COLORS[p], linewidth=2)
        plt.scatter(backgrounds[p][:700, i], phis[p][:700, i], alpha=0.75, c=COLORS[p])
    plt.xlabel(feature_names_latex[i])
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.ylim(-1, 1)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.savefig(os.path.join(image_path, f"Attrib_{i}_regional.pdf"), bbox_inches='tight')
plot_legend(rules, ncol=3)
plt.savefig(os.path.join(image_path, "Legend.pdf"), bbox_inches='tight', pad_inches=0)

# plt.show()