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
X, y, features, task = setup_data_trees("bike")
d = len(features)
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
model, perf = load_trees("bike", "rf", 0)

# %%
# Uniform Background
background = x_train[:200]
Phis, _ = interventional_taylor_treeshap(model, background, background)

# %%
interactions_heatmap(Phis, features)
plt.show()

# %% HOUR versus YEAR
plot_interaction(2, 0, background, Phis, features)
plt.show()

# %% HOUR versus WORKINGDAY 
plot_interaction(2, 5, background, Phis, features)
plt.show()

# %% HOUR versus TEMPERATURE
plot_interaction(2, 7, background, Phis, features)
plt.show()

# %% [markdown]
# The strongest interactions involve features Hour, Temperature, 
# Year and WorkingDay. These four features will be used when fitting a FDTree.
# %%
