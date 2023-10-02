""" Find feature interactions in Kin8nm """
# %%
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_data_trees, custom_train_test_split, setup_pyplot_font
from utils import load_trees,  plot_interaction, interactions_heatmap

setup_pyplot_font(20)

sys.path.append(os.path.abspath(".."))
from src.anova import interventional_taylor_treeshap

# %%
# Load data and model
X, y, features, task = setup_data_trees("kin8nm")
x_train, x_test, y_train, y_test = custom_train_test_split(X, y, task)
# Load models
model_name = "gbt"
model, perf = load_trees("kin8nm", model_name, 0)

# %%
# Uniform Background
background = x_train[:500]
Phis, _ = interventional_taylor_treeshap(model, background, background)

# %%
interactions_heatmap(Phis, features)
plt.savefig(os.path.join("Images", "kin8nm", f"Interactions_{model_name}.pdf"), 
                                                            bbox_inches='tight')
plt.show()

# %% Theta4 and Theta7
plot_interaction(3, 6, background, Phis, features)
plt.savefig(os.path.join("Images", "kin8nm", f"Interactions_4_7.pdf"), 
                                                bbox_inches='tight')
plt.show()

# %% Theta4 and Theta6
plot_interaction(3, 5, background, Phis, features)
plt.savefig(os.path.join("Images", "kin8nm", f"Interactions_4_6.pdf"), 
                                                bbox_inches='tight')
plt.show()

# %% Theta3 and Theta5
plot_interaction(2, 4, background, Phis, features)
plt.savefig(os.path.join("Images", "kin8nm", f"Interactions_3_5.pdf"), 
                                                bbox_inches='tight')
plt.show()

# %% Theta6 and Theta7
plot_interaction(5, 6, background, Phis, features)
plt.savefig(os.path.join("Images", "kin8nm", f"Interactions_6_7.pdf"), 
                                                bbox_inches='tight')
plt.show()

# %% Theta5 and Theta6
plot_interaction(4, 5, background, Phis, features)
plt.savefig(os.path.join("Images", "kin8nm", f"Interactions_5_6.pdf"), 
                                                bbox_inches='tight')
plt.show()

# %% [markdown]
# Any angle interacts with others so we will use all
# features when fitting the FD-Tree
# %%
