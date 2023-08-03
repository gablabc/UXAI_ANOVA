import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import os

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.stats import gaussian_kde, mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

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
    



import sklearn.ensemble as se
import numpy as np


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
    if task == "regression":
        return train_test_split(X, y, test_size=0.1, random_state=42)
    else:
        return train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)



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
    
    if len(features.nominal) > 0:
        # One Hot Encoding
        ohe = ColumnTransformer([
                              ('id', FunctionTransformer(), features.non_nominal),
                              ('ohe', OneHotEncoder(sparse=False), features.nominal)])
        ohe.fit(X)
    else:
        ohe = None
        
    return X, y, features, task, ohe



def load_trees(name, model_name, reject=False):
    
    file_path = os.path.join("models", name, model_name)
    
    # Pickle model
    from joblib import load
    models_files = [f for f in os.listdir(file_path) if "joblib" in f]
    # Sort by seed value
    models_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # print(models_files)
    models = []
    perfs = []
    for model_file in models_files:
        seed = int(model_file.split("_")[2].split(".")[0])
        models.append(load(os.path.join(file_path, model_file)))
        if model_name == "rf":
            models[-1].set_params(n_jobs=1)
        perf = pd.read_csv(os.path.join(file_path, f"perfs_seed_{seed}.csv")).to_numpy()
        perfs.append(perf)      
    perfs = np.array(perfs).squeeze()
    # if reject:
    #     # Get performances
    #     threshold = THRESHOLDS_MAPPING[name]
    #     good_model_idx = np.where(perfs[:, 1] < threshold)[0]
    #     if len(good_model_idx) < len(perfs):
    #         print("Some models are very bad !!!")
    #     return [models[i] for i in good_model_idx], perfs[good_model_idx]
    # else:
    return models, perfs        



def save_tree(model, args, perf_df):
    # Make folder for dataset models
    folder_path = os.path.join("models", args.data.name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_path = os.path.join(folder_path, args.model_name)
    # Make folder for architecture
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    state = args.ensemble.random_state
    # Pickle model
    from joblib import dump
    filename = f"random_state_{state}.joblib"
    dump(model, os.path.join(file_path, filename))

    # Save performance in CSV file
    perf_df.to_csv(os.path.join(file_path,f"perfs_seed_{state}.csv"), index=False)

    # Save model hyperparameters
    json.dump(model.get_params(), open(os.path.join(file_path,
                                       "hparams.json"), "w"), indent=4)



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




@dataclass
class TreeEnsembleHP:
    n_estimators: int = 100 # Number of trees in the forest
    max_depth: int = -1 # Maximal depth of each tree
    min_samples_leaf: int = 1 # No leafs can have less samples than this
    min_samples_split: int = 2 # Nodes with fewer samples cannot be splitS
    random_state: int = 0 # Random seed of the learning algorithm

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
class Config:
    save: bool = False # Save results of runs e.g. models, explainations ...


@dataclass
class Wandb_Config:
    wandb: bool = False  # Use wandb logging
    wandb_project: str = "UXAI"  # Which wandb project to use
    wandb_entity: str = "galabc"  # Which wandb entity to use


@dataclass
class Data_Config:
    name: str = "bike"  # Name of dataset "bike", "california", "boston"
    batch_size: int = 50  # Mini batch size
    scaler: str = "Standard"  # Scaler used for features and target


@dataclass
class Search_Config:
    n_splits: int = 5  # Number of train/valid splits
    split_type: str = "Shuffle" # Type of cross-valid "Shuffle" "K-fold"
    split_seed: int = 1 # Seed for the train/valid splits reproducability
