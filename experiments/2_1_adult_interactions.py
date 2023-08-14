""" Find feature interactions in Bike Sharing"""
# %%
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_data_trees, custom_train_test_split
from utils import load_trees

sys.path.append(os.path.abspath(".."))
from src.anova import interventional_taylor_treeshap

# %%
# Load data and model
X, y, features, task = setup_data_trees("adult_income")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
models, perfs = load_trees("adult_income", "rf")

# %%
# Uniform Background
background = x_train[:200]
Phis, _ = interventional_taylor_treeshap(models[0], background, background)

# %%
Phi_imp = np.abs(Phis).mean(0)
Phi_imp[np.abs(Phi_imp) < 2e-3] = 0

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(Phi_imp, cmap='Reds')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(12))
ax.set_xticklabels(features.names)
ax.set_yticks(np.arange(12))
ax.set_yticklabels(features.names)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                                rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(12):
    for j in range(12):
        text = ax.text(j, i, f"{Phi_imp[i, j]:.3f}",
                       ha="center", va="center", color="w")

ax.set_title("Shapley-Taylor Global indices")
fig.tight_layout()
plt.show()

# %% 
# Visualize the strongest interactions
def plot_interaction(i, j):
    plt.figure()
    plt.scatter(background[:, i], 
                background[:, j], c=2*Phis[:, i, j], 
                cmap='seismic', alpha=0.75)
    plt.xlabel(features.names[i])
    plt.ylabel(features.names[j])
    if features.types[i] == "ordinal":
        plt.xticks(np.arange(len(features.maps[i].cats)),
                   features.maps[i].cats)
    if features.types[j] == "ordinal":
       plt.yticks(np.arange(len(features.maps[j].cats)),
                   features.maps[j].cats)
    plt.colorbar()
    plt.show()

# %% AGE vs RELATIONSHIP
plot_interaction(0, 10)

# %% HOURS vs RELATIONSHIP
plot_interaction(4, 10)

# %% MARITAL-STATUS vs RELATIONSHIP
plot_interaction(8, 10)

# %% [markdown]
# The strongest interactions involve features AGE RELATIONSHIP
# HOURS and MARITAL-STATUS. These four features will be used 
# when fitting a FDTree.
# %%
