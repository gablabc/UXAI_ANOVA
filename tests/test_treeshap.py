""" Assert that TreeSHAP returns the right values """

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from scipy.stats import chi2
import time

import shap
from shap.explainers import Tree, Exact
from shap.maskers import Independent

import sys, os
sys.path.append(os.path.join(".."))
from src.anova import interventional_treeshap, interventional_taylor_treeshap
from src.anova import get_ANOVA_1, interventional_additive_treeshap
from src.anova import get_A_treeshap


def compare_shap_implementations(X, model, black_box):
    if X.shape[0] > 1000:
        X = X[:1000]
    # Run the original treeshap
    background = X[:100]
    masker = Independent(background, max_samples=100)
    explainer = Exact(black_box, masker=masker)
    orig_shap = explainer(X).values

    # Run the custom treeshap
    custom_shap, _ = interventional_treeshap(model, X, background)

    # Make sure we output the same result
    assert np.isclose(orig_shap, custom_shap).all()



def check_shap_additivity(X, model, I_map, task="regression"):
    if X.shape[0] > 1000:
        X = X[:1000]
    background = X[:100]

    # Prediction Gaps
    if task == "regression":
        gaps = model.predict(X) - model.predict(background).mean()
    else:
        gaps = model.predict_proba(X)[:, 1] - model.predict_proba(background)[:, 1].mean()
    
    # Run the custom treeshap
    custom_shap, _ = interventional_treeshap(model, X, background, I_map=I_map)

    assert custom_shap.shape[1] == I_map[-1]+1, "Not one SHAP value per coallition"

    # Make sure the SHAP values add up to the gaps
    assert np.isclose(gaps, custom_shap.sum(1)).all()



def compare_shap_taylor_implementations(X, model, black_box):
    if X.shape[0] > 100:
        X = X[:100]
    # Run the original treeshap
    background = X[:100]
    masker = Independent(background, max_samples=100)
    explainer = Exact(black_box, masker=masker)
    orig_shap_taylor = explainer(X, interactions=2).values

    # Run the custom treeshap
    custom_shap_taylor, _ = interventional_taylor_treeshap(model, X, background)

    # Make sure we output the same result
    assert np.isclose(orig_shap_taylor, custom_shap_taylor).all()

    # Make sure that shap_taylor sums up to shap
    custom_shap, _ = interventional_treeshap(model, X, background)
    assert np.isclose(custom_shap_taylor.sum(-1), custom_shap).all()



def compare_anova1_implementations(X, model, black_box):
    if X.shape[0] > 500:
        X = X[:500]
    
    # Run the brute-force method
    A = get_ANOVA_1(X, black_box)

    # Run the custom treeshap
    start = time.time()
    fast_A = interventional_additive_treeshap(model, X)
    end = time.time()
    print(f"Custom Anova1 took {end-start:.1f} seconds")

    # Make sure we output the same result
    assert np.isclose(A[..., 1:], fast_A).all()



def compare_A_implementations(X, model, black_box, use_stack):
    if X.shape[0] > 100:
        X = X[:100]
    
    # Run the brute-force method
    A = get_ANOVA_1(X, black_box).sum(-1)

    # Run the custom treeshap
    start = time.time()
    fast_A = get_A_treeshap(model, X, use_stack)
    end = time.time()
    print(f"Custom A took {end-start:.1f} seconds")

    # Make sure we output the same result
    assert np.isclose(A, fast_A).all()




def setup_task(d, correlations, task):
    np.random.seed(42)
    # Generate input
    if correlations:
        mu = np.zeros(d)
        sigma = 0.5 * np.eye(d) + 0.5 * np.ones((d, d))
        X = np.random.multivariate_normal(mean=mu, cov=sigma, size=(1000,))
    else:
        X = np.random.normal(0, 1, size=(1000, d))
    
    # Generate target and fit model
    if task == "regression":
        y = X.mean(1)
        model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
    else:
        y = (np.linalg.norm(X, axis=1) > np.sqrt(chi2(df=d).ppf(0.5))).astype(int)
        model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
    if task == "regression":
        black_box = model.predict
    else:
        black_box = lambda x : model.predict_proba(x)[:, -1]

    return X, y, model, black_box



####### TreeSHAP on toy data ########
@pytest.mark.parametrize("d", range(2, 13))
@pytest.mark.parametrize("correlations", [False, True])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("interactions", [1, 2])
def test_treeshap_implementation(d, correlations, task, interactions):

    # Setup data and model
    X, y, model, black_box = setup_task(d, correlations, task)

    # Compute SHAP values
    if interactions == 1:
        compare_shap_implementations(X, model, black_box)
    else:
        compare_shap_taylor_implementations(X, model, black_box)





####### ANOVA 1 Projection ########
@pytest.mark.parametrize("d", range(2, 13))
@pytest.mark.parametrize("correlations", [False, True])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_anova1_implementation(d, correlations, task):

    # Setup data and model
    X, y, model, black_box = setup_task(d, correlations, task)
    
    # Run test
    compare_anova1_implementations(X, model, black_box)




####### A Matrix computation ########
@pytest.mark.parametrize("d", range(2, 22, 2))
@pytest.mark.parametrize("correlations", [False, True])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("use_stack", [False, True])
def test_A_implementation(d, correlations, task, use_stack):

    # Setup data and model
    X, y, model, black_box = setup_task(d, correlations, task)
    
    # Run test
    compare_A_implementations(X, model, black_box, use_stack)





# @pytest.mark.parametrize("d", range(4, 21, 4))
# def test_treeshap_regression_coallition(d):
#     # Generate data
#     X = np.random.normal(0, 1, size=(1000, d))
#     y = X.mean(1)

#     # Determine coallitions
#     n_coallitions = d // 4
#     coallition_size = int(d / n_coallitions)
#     I_map = np.ravel([[i]*coallition_size for i in range(n_coallitions)])

#     # Fit model
#     model = RandomForestRegressor(n_estimators=40, max_depth=10).fit(X, y)

#     # Compute SHAP values
#     check_shap_additivity(X, model, I_map)



# @pytest.mark.parametrize("d", range(4, 21, 4))
# def test_treeshap_classification_coallition(d):
#     np.random.seed(42)

#     # Generate data
#     X = np.random.normal(0, 1, size=(1000, d))
#     y = (np.linalg.norm(X, axis=1) > np.sqrt(chi2(df=d).ppf(0.5))).astype(int)

#     # Determine coallitions
#     n_coallitions = d // 4
#     coallition_size = int(d / n_coallitions)
#     I_map = np.ravel([[i]*coallition_size for i in range(n_coallitions)])

#     # Fit model
#     model = RandomForestClassifier(n_estimators=40, max_depth=10).fit(X, y)

#     # Compute SHAP values
#     check_shap_additivity(X, model, I_map, task="classification")





# # Test with adult without one-hot-encoding
# def test_adult_no_ohe():
#     # Get data without OHE directly from SHAP library
#     X, y = shap.datasets.adult()

#     # Fit model
#     model = RandomForestClassifier(random_state=23, n_estimators=50, 
#                                    max_depth=4, min_samples_leaf=50, n_jobs=-1)
#     model.fit(X, y)

#     # Compute SHAP values
#     compare_shap_implementations(X, model)


# # Test with adult with one-hot-encoding
# def test_adult_ohe():
#     import sys, os
#     sys.path.append(os.path.join("..", "..", "experiments"))
#     from utils import setup_data_trees

#     X, y, _, task, ohe, I_map = setup_data_trees("adult_income")
#     # Encode for training
#     X = ohe.transform(X)

#     # Fit model
#     model = RandomForestClassifier(random_state=23, n_estimators=50, 
#                                    max_depth=4, min_samples_leaf=50, n_jobs=-1)
#     model.fit(X, y)

#     # Compute SHAP values
#     check_shap_additivity(X, model, I_map=I_map, task=task)


if __name__ == "__main__":
    # for d in range(10, 60, 10):
    #     test_anova1_implementation(d, False, "classification")
    # test_A_implementation(30, False, "regression", False)
    test_treeshap_implementation(3, False, "classification", interactions=2)
    # test_adult_no_ohe()

