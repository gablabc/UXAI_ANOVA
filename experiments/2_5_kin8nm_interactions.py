""" Find feature interactions in Kin8nm """
# %%
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_data_trees, custom_train_test_split
from utils import load_trees,  plot_interaction, interactions_heatmap

sys.path.append(os.path.abspath(".."))
from src.anova import interventional_taylor_treeshap

# %%
# Load data and model
X, y, features, task = setup_data_trees("kin8nm")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
model, perf = load_trees("kin8nm", "gbt", 0)

# %%
# Uniform Background
background = x_train[:500]
Phis, _ = interventional_taylor_treeshap(model, background, background)

# %%
interactions_heatmap(Phis, features)
plt.show()

# %% Theta4 and Theta7
plot_interaction(3, 6, background, Phis, features)
plt.show()

# %% Theta4 and Theta6
plot_interaction(3, 5, background, Phis, features)
plt.show()

# %% Theta3 and Theta5
plot_interaction(2, 4, background, Phis, features)
plt.show()
# %% [markdown]
# Any angle interacts with others so we will use all
# features when fitting the FD-Tree
# %%
