
import ctypes
import glob
import pandas as pd
import sklearn.ensemble as se
import numpy as np
from functools import partial
from tqdm import tqdm
import os
from shap.explainers import Tree



def get_ANOVA_1(X, f):
    N, D = X.shape
    data_temp = np.copy(X)
    H = np.zeros((N, N, D+1))
    f_X = f(X)
    H[..., 0] += f_X.reshape((1, -1))
    H[..., 1:] -= f_X.reshape((1, -1, 1))
    for d in tqdm(range(D), desc="ANOVA-1"):
        for n in range(N):
            data_temp[:, d] = X[n, d]
            H[n, :, d+1] += f(data_temp)
        # Reset
        data_temp[:, d] = X[:, d] 
    
    # Sanity Checks : Diagonal elements should be equal to f(x)
    assert np.isclose(H.sum(-1)[np.arange(N), np.arange(N)], f_X).all()
    return H


def get_ANOVA_1_tree(X, tree_ensemble, task, logit=False):
    # The black-box to call
    if task == "regression":
        f = tree_ensemble.predict
    else:
        if logit:
            f = tree_ensemble.decision_function
        else:
            f = lambda x : tree_ensemble.predict_proba(x)[:, 1]
    
    N, D = X.shape
    H = np.zeros((N, N, D+1))
    f_X = f(X)
    H[..., 0] += f_X.reshape((1, -1))
    H[..., 1:] = interventional_additive_treeshap(tree_ensemble, X)
    
    # Sanity Checks : Diagonal elements should be equal to f(x)
    assert np.isclose(H.sum(-1)[np.arange(N), np.arange(N)], f_X).all()
    return H


def get_ANOVA_2(X, f, features):
    assert len(features) == 2
    N, _ = X.shape
    data_temp = np.copy(X)
    F = np.zeros((N, N, 3))
    f_X = f(X)
    F -= f_X.reshape((1, -1, 1))
    k = 0
    # Additive terms
    for d in tqdm(range(2), desc="ANOVA-2"):
        for n in range(N):
            data_temp = np.copy(X)
            data_temp[:, features[d]] = X[n, features[d]]
            F[n, :, k] += f(data_temp)
        # Reset
        data_temp[:, features[d]] = X[:, features[d]] 
        k += 1
    # Interaction
    for n in tqdm(range(N)):
        data_temp[:, features] = X[n, features]
        F[n, :, -1] += f(data_temp)
        F[n, :, -1] -= F[n, :, :-1].sum(-1)
    return F


def get_PFI(H):
    N, _, D = H[..., 1:].shape
    E_remove_i = np.zeros((N, D))
    for d in tqdm(range(D), desc="Permutation Feature Importance"):
        E_remove_i[:, d] = H[..., d+1].mean(0)
    I = np.mean(E_remove_i**2, axis=0)
    return I



def interventional_treeshap(model, foreground, background, I_map=None):
    """ 
    Compute the Interventional Shapley Values with the TreeSHAP algorithm

    Parameters
    ----------
    model : model_object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost, Pyspark
        and most tree-based scikit-learn models are supported.

    foreground : numpy.array or pandas.DataFrame
        The foreground dataset is the set of all points whose prediction we wish to explain.

    background : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out missing features in the coallitional game.

    I_map : List(int), default=None
        A mapping from column to high-level feature. This is useful when feature are one-hot-encoded
        but you really want a single Shapley value for each group of columns. For example,
        `I_map = [0, 1, 2, 2, 2]` treats the last three columns as an encoding of the same feature. 
        Therefore we would return 3 Shapley values. Setting `I_map`
        to None will yield one Shapley value per column.
    """

    # Extract tree structure with the SHAP API
    ensemble = Tree(model, data=background).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    # All arrays must be C-Contiguous and DataFrames are not.
    if type(foreground) == pd.DataFrame:
        foreground = np.ascontiguousarray(foreground)
    if type(background) == pd.DataFrame:
        background = np.ascontiguousarray(background)

    # Mapping from column to partition index
    if I_map is None:
        I_map = np.arange(foreground.shape[1]).astype(np.int32)
    else:
        I_map = I_map.astype(np.int32)
    
    # Shapes
    Nt = ensemble.features.shape[0]
    n_features = np.max(I_map) + 1
    depth = ensemble.features.shape[1]
    Nx = foreground.shape[0]
    Nz = background.shape[0]

    # Values at each leaf
    values = np.ascontiguousarray(ensemble.values[..., -1])

    # Where to store the output
    results = np.zeros((Nx, n_features))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version    
    project_root = os.path.dirname(__file__).split('src')[0]
    libfile = glob.glob(os.path.join(project_root, 'build', '*', 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_int_treeshap.restype = ctypes.c_int
    mylib.main_int_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                        ctypes.c_int, ctypes.c_int,
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.float64),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.int32),
                                        np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_int_treeshap(Nx, Nz, Nt, foreground.shape[1], depth, foreground, background, 
                            I_map, ensemble.thresholds, values,
                            ensemble.features, ensemble.children_left, 
                            ensemble.children_right, results)

    return results, ensemble




def interventional_taylor_treeshap(model, foreground, background):
    
    # Extract tree structure with the SHAP API
    ensemble = Tree(model, data=background).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    values = np.ascontiguousarray(ensemble.values[..., -1])
    if type(foreground) == pd.DataFrame:
        foreground = np.ascontiguousarray(foreground).astype(np.float64)
    if type(background) == pd.DataFrame:
        background = np.ascontiguousarray(background).astype(np.float64)
    foreground = foreground.astype(np.float64)
    background = background.astype(np.float64)

    # Shape properties
    Nx = foreground.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]
    d = foreground.shape[1]
    depth = ensemble.features.shape[1]

    # Where to store the output
    results = np.zeros((Nx, d, d))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version    
    project_root = os.path.dirname(__file__).split('src')[0]
    libfile = glob.glob(os.path.join(project_root, 'build', '*', 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_taylor_treeshap.restype = ctypes.c_int
    mylib.main_taylor_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_taylor_treeshap(Nx, Nz, Nt, d, depth, foreground, background, 
                                ensemble.thresholds, values,
                                ensemble.features, ensemble.children_left, 
                                ensemble.children_right, results)

    return results, ensemble





def interventional_additive_treeshap(model, X):
    
    # Extract tree structure with the SHAP API
    ensemble = Tree(model, data=X).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    values = np.ascontiguousarray(ensemble.values[..., -1])
    if type(X) == pd.DataFrame:
        X = np.ascontiguousarray(X).astype(np.float64)
    X = X.astype(np.float64)

    # Shape properties
    N, d = X.shape
    Nt = ensemble.features.shape[0]
    depth = ensemble.features.shape[1]

    # Where to store the output
    results = np.zeros((N, N, d))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version    
    project_root = os.path.dirname(__file__).split('src')[0]
    libfile = glob.glob(os.path.join(project_root, 'build', '*', 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_additive_treeshap.restype = ctypes.c_int
    mylib.main_additive_treeshap.argtypes = [ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_additive_treeshap(N, Nt, d, depth, X,
                                ensemble.thresholds, values,
                                ensemble.features, ensemble.children_left, 
                                ensemble.children_right, results)

    return results





def get_Hadd__treeshap(model, X, use_stack=False):
    
    # Extract tree structure with the SHAP API
    ensemble = Tree(model, data=X).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    values = np.ascontiguousarray(ensemble.values[..., -1])
    if type(X) == pd.DataFrame:
        X = np.ascontiguousarray(X).astype(np.float64)
    X = X.astype(np.float64)

    # Shape properties
    N, d = X.shape
    Nt = ensemble.features.shape[0]
    depth = ensemble.features.shape[1]

    # Where to store the output
    results = np.zeros((N, N))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version    
    project_root = os.path.dirname(__file__).split('src')[0]
    libfile = glob.glob(os.path.join(project_root, 'build', '*', 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_A_treeshap.restype = ctypes.c_int
    mylib.main_A_treeshap.argtypes = [ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    ctypes.c_bool]

    # 3. call function mysum
    mylib.main_A_treeshap(N, Nt, d, depth, X,
                                ensemble.thresholds, values,
                                ensemble.features, ensemble.children_left, 
                                ensemble.children_right, results, use_stack)
    results += ensemble.base_offset[-1]
    return results