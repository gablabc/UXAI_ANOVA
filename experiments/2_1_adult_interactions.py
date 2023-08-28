""" Find feature interactions in Bike Sharing"""
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
X, y, features, task = setup_data_trees("adult_income")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
model, perf = load_trees("adult_income", "rf", 0)

# %%
# Uniform Background
background = x_train[:200]
Phis, _ = interventional_taylor_treeshap(model, background, background)

# %%
interactions_heatmap(Phis, features)
plt.show()

# %% AGE vs RELATIONSHIP
plot_interaction(0, 10, background, Phis, features)
plt.show()

# %% HOURS vs RELATIONSHIP
plot_interaction(4, 10, background, Phis, features)
plt.show()

# %% MARITAL-STATUS vs RELATIONSHIP
plot_interaction(8, 10, background, Phis, features)
plt.show()

# %% [markdown]
# The strongest interactions involve features AGE RELATIONSHIP
# HOURS and MARITAL-STATUS. These four features will be used 
# when fitting a FDTree.
# %%
