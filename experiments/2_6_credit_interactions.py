""" Find feature interactions in Default Credit """
# %%
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_data_trees, custom_train_test_split, setup_pyplot_font
from utils import load_trees, plot_interaction, interactions_heatmap

setup_pyplot_font(25)

sys.path.append(os.path.abspath(".."))
from src.anova import interventional_taylor_treeshap

# %%
# Load data and model
X, y, features, task = setup_data_trees("default_credit")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
model_name = "rf"
model, perf = load_trees("default_credit", model_name, 0)

# %%
# Uniform Background
background = x_train[:200]
Phis, _ = interventional_taylor_treeshap(model, background, background)

# %%
interactions_heatmap(Phis, features)
plt.savefig(os.path.join("Images", "default_credit", f"Interactions_{model_name}.pdf"), 
                                                            bbox_inches='tight')
plt.show()

# %% 
plot_interaction(2, 3, background, Phis, features)
plt.show()

# %% 
plot_interaction(2, 8, background, Phis, features)
plt.show()

# %% 
plot_interaction(8, 9, background, Phis, features)
plt.show()

# %%
