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
X, y, features, task, ohe = setup_data_trees("bike")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
models, perfs = load_trees("bike", "rf")

# %%
# Uniform Background
background = x_train[:200]
Phis, _ = interventional_taylor_treeshap(models[0], background, background)

# %%
Phi_imp = np.abs(Phis).mean(0)
Phi_imp[np.abs(Phi_imp) < 2e-3] = 0

fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(Phi_imp, cmap='Reds')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(10))
ax.set_xticklabels(features.names)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(features.names)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(10):
    for j in range(10):
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
    plt.colorbar()
    plt.show()

# %% HOUR versus YEAR
plot_interaction(2, 0)

# %% HOUR versus WORKINGDAY 
plot_interaction(2, 5)

# %% HOUR versus TEMPERATURE
plot_interaction(2, 7)

# %% [markdown]
# The strongest interactions involve features Hour, Temperature, 
# Year and WorkingDay. These four features will be used when fitting a FDTree.
# %%
