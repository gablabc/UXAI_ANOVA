""" First example in the paper """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import shap
from shap.maskers import Independent

# Local
from utils import setup_pyplot_font, three_bars, plot_legend, COLORS
sys.path.append(os.path.abspath(".."))
from src.features import Features
from src.anova_tree import PARTITION_CLASSES
from src.anova import get_ANOVA_1, get_PFI

setup_pyplot_font(30)

image_path = os.path.join("Images", "Motivation")
# Generate the data
np.random.seed(42)
d = 5
latex_feature_names = [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
X = np.random.uniform(-1, 1, size=(1000, d))
features = Features(X, latex_feature_names, ["num"]*d)
def h(X):
    y_hat = np.zeros((X.shape[0]))
    mask = (X[:, 1] > 0)
    y_hat[mask] = X[mask, 0]
    y_hat[~mask] = X[~mask, 2]
    return y_hat
y = h(X)


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
for i in range(3):
    plt.figure()
    sorted_idx = np.argsort(background[:, i])
    plt.plot(background[sorted_idx, i], A[..., i+1].mean(1)[sorted_idx], 'k-')
    plt.scatter(background[:, i], phis[:, i], alpha=0.5)
    plt.xlabel(latex_feature_names[i])
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.ylim(-1, 1)
    plt.savefig(os.path.join(image_path, f"attrib_feature_{i}.pdf"), bbox_inches='tight')

# Global feature importance
I_PDP = np.var(A[..., 1:].mean(axis=1), axis=0)
I_SHAP = (phis**2).mean(axis=0)
I_PFI = get_PFI(A)
three_bars(I_PFI, I_SHAP, I_PDP, features)
plt.gca().invert_yaxis()
plt.yticks(fontsize=35)
plt.xlabel("Feature Importance")
plt.savefig(os.path.join(image_path, f"Importance.pdf"), bbox_inches='tight')



############# Anova-Tree #############


# Fit the tree
partition_type = "l2coe"
# partition_type = "pfi"
# partition_type = "gadget-pdp"
tree = PARTITION_CLASSES[partition_type](features, max_depth=1, save_losses=True)
if partition_type == "l2coe":
    tree.fit(X, A.sum(-1))
elif partition_type == "pfi":
    tree.fit(X, A)
elif partition_type == "gadget-pdp":
    tree.fit(X, A)

# Print results
tree.print(verbose=True)
groups, rules = tree.predict(X, latex_rules=True)
print(rules)

# Plot the objective values w.r.t the split candidates
plt.figure()
for i in range(3):
    splits = tree.root.splits[i]
    objectives = tree.root.objectives[i]
    plt.plot(splits, objectives, '-o', label=latex_feature_names[i])
plt.ylim(0, y.var())
plt.xlabel(r"Split threshold $\gamma$")
if partition_type == "l2coe":
    plt.ylabel(r"$L_2$ Cost of Exclusion")
elif partition_type == "pfi":
    plt.ylabel("Loss PFI vs PDP")
elif partition_type == "gadget-pdp":
    plt.ylabel("Loss ICE vs PDP")
plt.legend()
plt.savefig(os.path.join(image_path, f"Loss.pdf"), bbox_inches='tight')



# Rerun SHAP and PDP
phis = [0] * tree.n_groups
backgrounds = [0] * tree.n_groups
for group_idx in range(tree.n_groups):
    idx_select = (groups == group_idx)
    background = X[idx_select]
    backgrounds[group_idx] = background

    # SHAP
    mu = h(background).mean()
    masker = Independent(background, max_samples=background.shape[0])
    explainer = shap.explainers.Exact(h, masker)
    phis[group_idx] = explainer(background).values

    # Reshape idx to index the F matrix
    idx_select = np.where(idx_select)[0].reshape((-1, 1))

    # Global Feature Importance
    I_PDP = np.var(A[..., 1:][idx_select, idx_select.T].mean(axis=1), axis=0)
    I_SHAP = (phis[group_idx]**2).mean(axis=0)
    I_PFI = get_PFI(A[idx_select, idx_select.T])
    three_bars(I_PFI, I_SHAP, I_PDP, features, color=COLORS[group_idx])
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=35)
    plt.xlabel("Feature Importance")
    plt.savefig(os.path.join(image_path, f"Importance_region_{group_idx}.pdf"), 
                                                            bbox_inches='tight')


# Compare SHAP and PDP on each leaf
for i in range(3):
    plt.figure()
    for p in range(tree.n_groups):
        plt.scatter(backgrounds[p][:, i], phis[p][:, i], alpha=0.5, 
                    c=COLORS[p], label=rules[p])
    legend_labels = plt.gca().get_legend_handles_labels()
    plt.xlabel(latex_feature_names[i])
    plt.ylabel(r"$\phi_" + str(i) + r"(\bm{x})$")
    plt.ylim(-1, 1)
    plt.savefig(os.path.join(image_path, f"attrib_feature_{i}_regions.pdf"), 
                                                            bbox_inches='tight')

plot_legend(rules, ncol=2)
filename = f"Legend.pdf"
plt.savefig(os.path.join(image_path, "Legend.pdf"), bbox_inches='tight', pad_inches=0)

# plt.show()