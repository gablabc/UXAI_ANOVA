""" Toy Example with Interactions """

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import shap
from shap.maskers import Independent

# Local
from utils import setup_pyplot_font
sys.path.append(os.path.abspath(".."))
from src.features import Features
from src.anova_tree import L2CoETree
from src.anova import get_ANOVA_1, get_ANOVA_2

setup_pyplot_font(20)

# Generate the data
np.random.seed(42)
feature_names = ["x0", "x1", "x2"]
X = np.random.uniform(-1, 1, size=(500, 3))
features = Features(X, [f"x{i}" for i in range(3)], ["num"]*3)
def h(X):
    return X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1]
y = h(X)


#### ANOVA Order-2 ####
F = get_ANOVA_2(X, h, [0, 1])

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(X[:, 0], F[..., 0].mean(-1), c='k')
axs[0, 0].set_xlabel('X0')
axs[0, 0].set_ylabel('Main effect')
axs[0, 1].scatter(X[:, 0], X[:, 1], c=F[..., 2].mean(-1), cmap='seismic', 
                  alpha=0.75, norm=TwoSlopeNorm(0))
axs[0, 1].set_xlabel('X1')
axs[0, 1].set_ylabel('X0')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].scatter(X[:, 1], F[..., 1].mean(-1), c='k')
axs[1, 1].set_xlabel('X1')
axs[1, 1].set_ylabel('Main effect')


#### Taylor-SHAP ####
masker = Independent(X, max_samples=X.shape[0])
explainer = shap.explainers.Exact(h, masker=masker)
shap_taylor = explainer(X, interactions=2).values

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(X[:, 0], shap_taylor[:, 0, 0], c='k')
axs[0, 0].set_xlabel('X0')
axs[0, 0].set_ylabel('Main effect')
axs[0, 1].scatter(X[:, 0], X[:, 1], c=2*shap_taylor[:, 0, 1], cmap='seismic', 
                                            alpha=0.75, norm=TwoSlopeNorm(0))
axs[0, 1].set_xlabel('X1')
axs[0, 1].set_ylabel('X0')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1,0]')
axs[1, 1].scatter(X[:, 1], shap_taylor[:, 1, 1], c='k')
axs[1, 1].set_xlabel('X1')
axs[1, 1].set_ylabel('Main effect')


#### Fit ANOVA-Tree ####
A = get_ANOVA_1(X, h)
tree = L2CoETree(features, max_depth=1, save_losses=True)
tree.fit(X, A.sum(-1))
tree.print(verbose=True)

plt.show()