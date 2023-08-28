""" Find feature interactions in California Housing """
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
X, y, features, task = setup_data_trees("california")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
model, perf = load_trees("california", "gbt", 0)

# %%
# Uniform Background
background = x_train[:500]
Phis, _ = interventional_taylor_treeshap(model, background, background)

# %%
interactions_heatmap(Phis, features)
plt.show()

# %% LATITUDE versus LONGITUDE
plot_interaction(6, 7, background, Phis, features)
plt.show()

# %% AGE versus OCCUP (Friedmann et al page 374)
plot_interaction(5, 1, background, Phis, features)
plt.show()

# %% [markdown]
# The strongest interactions involve the correlated LAT and LONG features
# Hence the FD-tree may split the state of California on the map.
# We also include AGE and OCCUP which interact but to a lesser extent.
# %%
