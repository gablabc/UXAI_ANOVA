import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import os

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.stats import gaussian_kde, mannwhitneyu, rankdata

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import sklearn.ensemble as se

from data_utils import DATASET_MAPPING, TASK_MAPPING


def setup_pyplot_font(size=11):
    from matplotlib import rc
    rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':size})
    rc('text', usetex=True)
    from matplotlib import rcParams
    rcParams["text.latex.preamble"] = r"\usepackage{bm}\usepackage{amsfonts}"



color_dict = {}
color_dict["default"] = {'zero' : [255, 255, 255], 
                         'pos':  [0, 102, 255], 
                         'neg' : [255, 69, 48]}
color_dict["DEEL"] = {'zero' : [255, 255, 255], 
                      'pos':  [0, 69, 138], 
                      'neg' : [255, 69, 48]}
COLORS = ['blue', 'red', 'green', 'orange',
          'violet', 'brown', 'cyan', 'olive']


def abs_map_CIs(phi, CIs):
    """
    Map CIs on phi to CIs on |phi|
    """
    FI = np.abs(phi)
    cross_origin = FI < CIs
    # Min and Max of CIs
    map_bottom = np.abs(phi - CIs)
    map_top = np.abs(phi + CIs)
    min_CIs = np.minimum(map_bottom, map_top)
    max_CIs = np.maximum(map_bottom, map_top)
    # Minimum of CIs that cross origin is zero
    min_CIs[cross_origin] = 0
    return min_CIs, max_CIs



def bar(phis, feature_labels, threshold=None, xerr=None, absolute=False, ax=None):

    # Are there multiple feature labels?
    if type(feature_labels[0]) == list:
        num_features = len(feature_labels[0])
        multiple_labels = True
    else:
        num_features = len(feature_labels)
        multiple_labels = False

    if absolute:
        bar_mapper = lambda x : np.abs(x)
        if xerr is not None:
            min_CIs, max_CIs = abs_map_CIs(phis, xerr)
            xerr = np.abs(np.abs(phis) - np.vstack((min_CIs, max_CIs)))
    else:
        bar_mapper = lambda x : x
        
    ordered_features = np.argsort(bar_mapper(phis))
    y_pos = np.arange(0, len(ordered_features))
        
    if ax is None:
        plt.figure()
        # plt.gcf().set_size_inches(16, 10)
        ax = plt.gca()
        
    negative_phis = (phis < 0).any() and not absolute
    if negative_phis:
        ax.axvline(0, 0, 1, color="k", linestyle="-", linewidth=1, zorder=1)
    if threshold:
        ax.axvline(threshold, 0, 1, color="k", linestyle="--", linewidth=2, zorder=1)
        if negative_phis:
            ax.axvline(-threshold, 0, 1, color="k", linestyle="--", linewidth=2, zorder=1)
    # draw the bars
    bar_width = 0.7
    
    # Get DEEL colors
    colors = deepcopy(color_dict["DEEL"])
    colors['pos'] = np.array(colors['pos'])/255.
    colors['neg'] = np.array(colors['neg'])/255.
    
    if xerr is not None:
        if xerr.ndim == 2 and xerr.shape[0] == 2:
            xerr = xerr[:, ordered_features]
        else:
            xerr = xerr[ordered_features]
    # Plot the bars with errr
    ax.barh(
        y_pos, bar_mapper(phis[ordered_features]),
        bar_width, xerr=xerr, align='center',
        color=[colors['neg'] if phis[ordered_features[j]] <= 0 
                else colors['pos'] for j in range(len(y_pos))], 
        edgecolor=(1,1,1,0.8), capsize=5
    )

    # Set the y-ticks and labels
    if multiple_labels:
        yticklabels = [feature_labels[0][j] for j in ordered_features]
        ax.set_yticks(ticks=list(y_pos))
        ax.set_yticklabels(yticklabels, fontsize=15)

        yticklabels = [feature_labels[1][j] for j in ordered_features]
        ax2 = ax.twinx()
        ax2.set_yticks(ticks=list(y_pos))
        ax2.set_yticklabels(yticklabels, fontsize=15)
        ax2.set_ybound(*ax.get_ybound())
    else:
        yticklabels = [feature_labels[j] for j in ordered_features]
        ax.set_yticks(ticks=list(y_pos))
        ax.set_yticklabels(yticklabels, fontsize=15)


    # put horizontal lines for each feature row
    for i in range(num_features):
        ax.axhline(i, color="k", lw=0.5, alpha=0.5, zorder=-1)

    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    
    if negative_phis:
        ax.set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        ax.set_xlim(xmin, xmax + (xmax-xmin)*0.05)
    
    plt.gcf().tight_layout()



def three_bars(data_1, data_2, data_3, features, color=None, sort=False):
    ind = np.arange(len(features))
    width = 0.25

    # Sort w.r.t the first data
    if sort:
        sorted_idx = np.argsort(data_1)
        feature_names = [features.names[i] for i in sorted_idx]
        data_1 = data_1[sorted_idx]
        data_2 = data_2[sorted_idx]
        data_3 = data_3[sorted_idx]
    else:
        feature_names = features.names
    # Default color is DEEL blue
    if color is None:
        color = np.array(color_dict["DEEL"]["pos"])/255
    
    # make the plots
    fig, ax = plt.subplots()
    ax.barh(ind, data_1, width, color=color)
    ax.barh(ind + width, data_2, width, color=color, alpha=0.5)
    ax.barh(ind + 2*width, data_3, width, color=color, alpha=0.25)
    ax.set_yticks(ind + width)  # position axis ticks
    ax.set_yticklabels(feature_names)  # set them to the names



def attrib_scatter_plot(backgrounds, pdps, phis, i, features, args):
    plt.figure()
    # We do NOT have a list of backgrounds
    if type(backgrounds) == np.ndarray:
        backgrounds = [backgrounds]
        pdps = [pdps]
        phis = [phis]
        colors=['#1f77b4']
    # We have a list of backgrounds
    else:
        colors = COLORS
    
    n_groups = len(backgrounds)
    for p in range(n_groups):
        # For ordinal features, we add a jitter to better see the points
        if features.types[i] == "ordinal":
            jitter = np.random.uniform(-0.1, 0.1, size=backgrounds[p].shape[0])
            plt.scatter(backgrounds[p][:, i]+jitter, phis[p][:, i], alpha=0.5, c=colors[p])
        else:
            plt.scatter(backgrounds[p][:, i], phis[p][:, i], alpha=0.5, c=colors[p])
        
        # Plot the PDP as a line
        sorted_idx = np.argsort(backgrounds[p][:, i])
        plt.plot(backgrounds[p][sorted_idx, i], pdps[p][sorted_idx, i], 'k-')

    # xticks labels depend on the type of feature
    if features.types[i] == "ordinal":
        categories = features.maps[i].cats
        # Truncate names if too long
        if len(categories) > 7:
            categories = [name[:3] for name in categories]
        plt.xticks(np.arange(len(categories)), categories, size=15)

    # For bike we have specific xticks
    if args.data.name == "bike":
        if i==0:
            plt.xticks(np.arange(0, 12, 1))
        if i == 2:
            plt.xticks(np.arange(0, 25, 2))
        # else:
        #     plt.xticks(np.arange(0, 45, 10))
    plt.grid('on', zorder=1)
    plt.xlabel(features.names[i])
    plt.ylabel(f"Attrib of {features.names[i]}")



def plot_legend(rules, figsize=(5, 0.6), ncol=4):
    # Plot the legend separately
    plt.figure(figsize=figsize)
    for p in range(len(rules)):
        plt.scatter(0, 0, alpha=0.5, c=COLORS[p], label=rules[p])
    plt.legend(loc='center', ncol=ncol, prop={"size": 10}, framealpha=1)
    plt.axis('off')



# Visualize the strongest interactions
def plot_interaction(i, j, background, Phis, features):
    plt.figure()
    if features.types[j] == "ordinal":
        for category_idx, category in enumerate(features.maps[j].cats):
            idx = background[:, j] == category_idx
            plt.scatter(background[idx, i],
                        Phis[idx, i, j], alpha=0.75, c=COLORS[category_idx],
                        label=f"{features.names[j]}={category}")
        plt.legend()
        plt.xlabel(features.names[i])
        plt.ylabel("Interaction")
    else:
        plt.scatter(background[:, i], 
                    background[:, j], c=2*Phis[:, i, j], 
                    cmap='seismic', alpha=0.75)
        plt.xlabel(features.names[i])
        plt.ylabel(features.names[j])
        plt.colorbar()
    if features.types[i] == "ordinal":
        plt.xticks(np.arange(len(features.maps[i].cats)),
                   features.maps[i].cats)
    # if features.types[j] == "ordinal":
    #    plt.yticks(np.arange(len(features.maps[j].cats)),
    #                features.maps[j].cats)


def interactions_heatmap(Phis, features):
    d = len(features)
    Phi_imp = np.abs(Phis).mean(0)
    Phi_imp[np.abs(Phi_imp) < 2e-3] = 0

    fig, ax = plt.subplots(figsize=(d, d))
    im = ax.imshow(Phi_imp, cmap='Reds')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(d))
    ax.set_xticklabels(features.names)
    ax.set_yticks(np.arange(d))
    ax.set_yticklabels(features.names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                                    rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(d):
        for j in range(d):
            text = ax.text(j, i, f"{Phi_imp[i, j]:.3f}",
                        ha="center", va="center", color="w")

    ax.set_title("Shapley-Taylor Global indices")
    fig.tight_layout()



def get_all_tree_preds(X, ensemble, task='regression'):
    """ 
    Return predictions of all trees in the tree ensemble

    Parameters
    ----------
    X: (n_samples, n_features) `np.array`
        Samples on which to predict
    ensemble: `sklearn class`
        A class from `sklearn.ensemble._forest` that represents
        a forest of trees.
    task: `str`, default='regression'
        The type of task: regression, clasification etc.

    Returns
    -------
    all_preds: (n_samples, len(ensemble)) `np.array`
        Predicted values on each sample for all trees.
    """
    if type(ensemble) not in [se._forest.RandomForestRegressor,
                              se._forest.RandomForestClassifier,
                              se._forest.ExtraTreesRegressor,
                              se._forest.ExtraTreesClassifier]:
        raise TypeError("The tree ensemble provided is not valid !!!")
        
    # This line is done in parallel in sklearn
    all_leafs = ensemble.apply(X) # (n_samples, n_estimators)
    all_preds = np.zeros((X.shape[0], len(ensemble)))

    # This loop could be parallelized
    for j, estimator in enumerate(ensemble.estimators_):
        if task == "regression":
            values = estimator.tree_.value.ravel()
            all_preds[:, j] = values[all_leafs[:, j]]
        elif task=="classification":
            values = estimator.tree_.value[:, 0]
            values = values / np.sum(values, axis=1, keepdims=True)
            all_preds[:, j] = values[all_leafs[:, j], 1]
        else:
            raise NotImplementedError()
    return all_preds




############################## General Utilities ##############################

# Custom train/test split for reproducability (random_state is always 42 !!!)
def custom_train_test_split(X, y, task):
    N = X.shape[0]
    ratio = 0.1 if N > 6000 else 0.2
    if task == "regression":
        return train_test_split(X, y, test_size=ratio, random_state=42)
    else:
        return train_test_split(X, y, test_size=ratio, random_state=42, stratify=y)



def get_cross_validator(k, task, split_seed, split_type):
    # Train / Test split and cross-validator. Dont look at the test yet...
    if task == "regression":
        if split_type == "Shuffle":
            cross_validator = ShuffleSplit(n_splits=k, 
                                           test_size=0.1,
                                           random_state=split_seed)
        elif split_type == "K-Fold":
            cross_validator = KFold(n_splits=k, shuffle=True, 
                                    random_state=split_seed)
        else:
            raise ValueError("Wrong type of cross-validator")
        
    # Binary Classification
    else:
        if split_type == "Shuffle":
            cross_validator = StratifiedShuffleSplit(n_splits=k, 
                                                     test_size=0.1, 
                                                     random_state=split_seed)
        elif split_type == "K-Fold":
            cross_validator = StratifiedKFold(n_splits=k,
                                              shuffle=True,
                                              random_state=split_seed)
        else:
            raise ValueError("Wrong type of cross-validator")
    
    return cross_validator



def get_hp_grid(filename):

    def to_eval(string):
        if type(string) == str:
            split = string.split("_")
            if len(split) == 2:
                return split[1]
            else:
                return None
        else:
            return None

    hp_dict = json.load(open(filename, "r"))
    for key, value in hp_dict.items():
        # Iterate over list
        if type(value) == list:
            for i, element in enumerate(value):
                str_to_eval = to_eval(element)
                if str_to_eval is not None:
                    value[i] = eval(str_to_eval)
        # Must be evaluated
        if type(value) == str:
            str_to_eval = to_eval(value)
            if str_to_eval is not None:
                hp_dict[key] = eval(str_to_eval)
    return hp_dict



def get_background(x, background_size, random_state):
    np.random.seed(random_state)
    idx_choose = np.random.choice(range(len(x)), background_size, replace=False)
    background = x[idx_choose]
    return background


############################## Tree-based models ##############################
TREES = {
         "rf" : {"regression": RandomForestRegressor(), 
                 "classification": RandomForestClassifier()
                 },
         "gbt" : {"regression": GradientBoostingRegressor(), 
                  "classification": GradientBoostingClassifier()
                 }
        }



def setup_data_trees(name):
    X, y, features = DATASET_MAPPING[name]()
    task = TASK_MAPPING[name]
    
    return X, y, features, task


def load_trees(dataset, model, random_state):
    # Random state used for fitting
    state = str(random_state)
    file_path = os.path.join("models", dataset, model+"_"+state)
    
    # Pickle model
    from joblib import load
    model = load(os.path.join(file_path, "model.joblib"))
    perf = pd.read_csv(os.path.join(file_path, f"performance.csv")).to_numpy()
    return model, perf


def save_tree(model, dataset, model_name, random_state, perf_df):
    # Random state used for fitting
    state = str(random_state)

    # Make folder for dataset models
    folder_path = os.path.join("models", dataset)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, model_name+"_"+state)
    # Make folder for architecture
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Pickle model
    from joblib import dump
    filename = f"model.joblib"
    dump(model, os.path.join(file_path, filename))

    # Save performance in CSV file
    perf_df.to_csv(os.path.join(file_path,f"performance.csv"), index=False)

    # Save model hyperparameters
    json.dump(model.get_params(), open(os.path.join(file_path,
                                       "hparams.json"), "w"), indent=4)



def save_FDTree(tree, dataset, model_name, random_state, partition_type, background_size):
    # Random state used for fitting
    state = str(random_state)

    # Make folder for dataset models
    folder_path = os.path.join("models", dataset)
    file_path = os.path.join(folder_path, model_name + "_" + state)
    
    # Pickle model
    from joblib import dump
    filename = f"{partition_type}_max_depth_{tree.max_depth}_N_{background_size}"
    dump(tree, os.path.join(file_path, filename + ".joblib"))

    # Save the print
    tree_string = tree.print(return_string=True, verbose=True)
    with open(os.path.join(file_path, filename + ".txt"), "w") as f:
        f.write(tree_string)



def load_FDTree(max_depth, dataset, model_name, random_state, partition_type, background_size):
    # Random state used for fitting
    state = str(random_state)

    # Make folder for dataset models
    folder_path = os.path.join("models", dataset)
    file_path = os.path.join(folder_path, model_name + "_" + state)
    
    # Pickle model
    from joblib import load
    filename = f"{partition_type}_max_depth_{max_depth}_N_{background_size}.joblib"
    tree = load(os.path.join(file_path, filename))

    return tree



######################### Explanation Disagreements ###########################

def rank_diff(phi_1, phi_2):
    rank_1 = rankdata(phi_1)
    rank_2 = rankdata(phi_2)
    return np.max(np.abs(rank_1 - rank_2))


def normalized_l2norm(phi_1, phi_2):
    phi_1 = phi_1 / phi_1.sum()
    phi_2 = phi_2 / phi_2.sum()
    return 100 * np.linalg.norm(phi_1-phi_2)


def pdp_vs_shap(pdp, shap):
    return np.sqrt(np.mean((pdp - shap)**2))
    # var = shap.var(0)
    # idx_non_zero = np.where(var > 0)[0]
    # error = np.mean((pdp - shap)**2, axis=0)[idx_non_zero]
    # var = var[idx_non_zero]
    # return 100 * np.mean(error / var)


############################## Distributional Shift ###########################

def U_tests_repeat(models, in_distr_var, ood_sampler, log_var, **kwargs):
    """ 
    Compute the log_var on multiple reruns of the explanation 
    distributions, show U-stats  p-vals with standard deviations and return
    the last run results
    """
        
    U_stats = np.array([0.0] * 10)
    p_vals  = np.array([0.0] * 10)
    # Repeat 10 times
    for i in range(10):
        ood_var = log_var(models, ood_sampler, **kwargs)
        # Significance test
        U_stats[i], p_vals[i] = \
            mannwhitneyu(ood_var, in_distr_var, alternative='greater')
    # Normalize the U stat
    U_stats /= (len(ood_var) * len(in_distr_var))
    print(f"U-stat : {U_stats.mean():.3f}  +- {U_stats.std():.3f} "+\
          f"with p-value {p_vals.mean():e} +- {p_vals.std():e}")
    
    return ood_var



def plot_hists(model_vars, labels):
    # Histogram
    plt.figure()
    cmap = plt.get_cmap("tab10")
    for i, log_var in enumerate(model_vars):
        plt.hist(log_var, bins=40, density=True, alpha=0.25,
                                                label=labels[i], color=cmap(i))
        xx = np.linspace(log_var.min(), log_var.max(), 100)
        plt.plot(xx, gaussian_kde(log_var).pdf(xx), color=cmap(i))
    plt.xlabel(r"$\log \Delta (\bm{x})$")
    plt.ylabel("Density")
    plt.legend(prop={'size': 11})
    

# from matplotlib import rc
# rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':15})
# rc('text', usetex=True)
# from matplotlib import rcParams
# rcParams['text.latex.preamble']=r"\usepackage{bm}\usepackage{amsfonts}"


def plot_var_hists(model_vars, labels, path=None):
    # Histogram
    fig, ax=plt.subplots()
    cmap = plt.get_cmap("tab10")
    for i, log_var in enumerate(model_vars):
        log_var=log_var
        ax.hist(log_var, bins=40, density=True, alpha=0.25,
                                                label=labels[i], color=cmap(i))
        xx = np.linspace(log_var.min(), log_var.max(), 100)
        ax.plot(xx, gaussian_kde(log_var).pdf(xx), color=cmap(i))
    ax.set_xlabel(r"$\log \Delta (\bm{x})$")
    ax.set_ylabel("Density")
    ax.legend()
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        #plt.show()
        #plt.close()  



##### Dataclasses for parsing arguments ####

@dataclass
class TreeEnsembleHP:
    n_estimators: int = 100 # Number of trees in the forest
    max_depth: int = -1 # Maximal depth of each tree
    min_samples_leaf: int = 1 # No leafs can have less samples than this
    min_samples_split: int = 2 # Nodes with fewer samples cannot be split
    max_features: float = 1.0 # Ratio of features to select at each split
    random_state: int = 2 # Random seed of the learning algorithm

@dataclass
class RandomForestHP():
    max_samples : float = 0.99 # Number of samples (relative to N) for boostrap
    criterion: str = "gini" # Split criterion, automatically set to MSE in regression


@dataclass
class GradientBoostingHP:
    subsample: float = 1.0 # Ratio of samples used to fit one tree (SGD)
    learning_rate: float = 0.1 # Learning rate of the algorithm
    n_iter_no_change: int = 50 # Early stopping, which will generate a valid set


@dataclass
class Wandb_Config:
    wandb: bool = False  # Use wandb logging
    wandb_project: str = "UXAI"  # Which wandb project to use
    wandb_entity: str = "galabc"  # Which wandb entity to use


@dataclass
class Data_Config:
    name: str = "california"  # Name of dataset "bike", "california", "boston"
    batch_size: int = 50  # Mini batch size
    scaler: str = "Standard"  # Scaler used for features and target


@dataclass
class Search_Config:
    n_splits: int = 5  # Number of train/valid splits
    split_type: str = "Shuffle" # Type of cross-valid "Shuffle" "K-fold"
    split_seed: int = 1 # Seed for the train/valid splits reproducability
