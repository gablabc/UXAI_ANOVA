""" Find feature interactions in Marketing """
# %%
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_data_trees, custom_train_test_split
from utils import load_trees, plot_interaction, interactions_heatmap

sys.path.append(os.path.abspath(".."))
from src.anova import interventional_taylor_treeshap

# %%
# Load data and model
X, y, features, task = setup_data_trees("marketing")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
model, perf = load_trees("marketing", "gbt", 0)

# %%
# Uniform Background
background = x_train[:500]
Phis, _ = interventional_taylor_treeshap(model, background, background)

# %%
interactions_heatmap(Phis, features)
plt.show()

# %% MONTH vs DAYS
plot_interaction(6, 5, background, Phis, features)
plt.show()

# %% MONTH vs CONTACT
plot_interaction(6, 14, background, Phis, features)
plt.show()

# %% MONTH vs PDAYS
plot_interaction(6, 9, background, Phis, features)
plt.show()

# %% [markdown]
# The strongest interactions involve features 
# MONTH, DAY, CONTACT, PDAYS
# %%
