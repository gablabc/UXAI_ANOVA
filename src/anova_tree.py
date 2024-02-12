from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from abc import ABC, abstractmethod



SYMBOLS = { True :
            {"leq" : "$\,\leq\,$",
            "and_str" : "$)\,\,\land$\,\,(",
            "up" : "$\,>\,$",
            "low" : "$\,<\,$",
            "in_set" : "$\in$"},
            False : 
            {"leq" : "≤",
            "and_str" : " & ",
            "up" : ">",
            "low": "<",
            "in_set" : "∈"}
        }


class Node(object):
    """ Node in a Decision Tree """
    def __init__(self, instances_idx, parent, depth, impurity):
        self.instances_idx = instances_idx
        self.parent = parent
        self.depth = depth
        self.impurity = impurity
        # Placeholders
        self.feature = None
        self.threshold = None
        self.child_left = None
        self.child_right = None
        self.splits = []
        self.objectives = []


    def update(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold




class FDTree(BaseEstimator, ABC):
    def __init__(self, features, 
                 max_depth=3, 
                 negligible_impurity=1, 
                 relative_decrease=0.7, 
                 samples_leaf=20, 
                 save_losses=False):
        self.features = features
        self.max_depth = max_depth
        self.negligible_impurity = negligible_impurity
        self.relative_decrease = relative_decrease
        self.samples_leaf = samples_leaf
        self.save_losses = save_losses


    def print(self, verbose=False, return_string=False):
        tree_strings = []
        self.recurse_print_tree_str(self.root, verbose=verbose, tree_strings=tree_strings)
        if return_string:
            return "\n".join(tree_strings)
        else:
            print("\n".join(tree_strings))

    
    def recurse_print_tree_str(self, node, verbose=False, tree_strings=[]):
        if verbose:
            tree_strings.append("|   " * node.depth + f"Loss {node.impurity:.4f}")
            tree_strings.append("|   " * node.depth + f"Samples {len(node.instances_idx):d}")
        # Leaf
        if node.child_left is None:
            tree_strings.append("|   " * node.depth + f"Group {node.group}")
        # Internal node
        else:
            curr_feature_name = self.features.names[node.feature]
            tree_strings.append("|   " * node.depth + f"If {curr_feature_name} ≤ {node.threshold:.4f}:")
            self.recurse_print_tree_str(node=node.child_left, verbose=verbose, tree_strings=tree_strings)
            tree_strings.append("|   " * node.depth + "else:")
            self.recurse_print_tree_str(node=node.child_right, verbose=verbose, tree_strings=tree_strings)


    def get_split_candidates(self, x_i, i):
        """ Return a list of split candiates """
        # Numerical features we take quantiles
        if self.features.types[i] == "num":
            if len(x_i) < 50:
                splits = np.quantile(x_i, [0.25, 0.5, 0.75])
            else:
                splits = np.quantile(x_i, np.arange(1, 10) / 10)
            # It is possible that quantiles equal the last element when there are
            # duplications. Hence we remove those splits to avoid leaves with no data
            splits = splits[~np.isclose(splits, np.max(x_i))]
        elif self.features.types[i] == "sparse_num":
            is_nonzero = np.where(x_i > 0)[0]
            if len(is_nonzero) == 0:
                splits = []
            else:
                x_i_nonzero = x_i[is_nonzero]
                if len(x_i_nonzero) < 50:
                    splits = np.append(0, np.quantile(x_i_nonzero, [0.25, 0.5, 0.75]))
                else:
                    splits = np.append(0, np.quantile(x_i_nonzero, np.arange(1, 10) / 10))
                # It is possible that quantiles equal the last element when there are
                # duplications. Hence we remove those splits to avoid leaves with no data
                splits = splits[~np.isclose(splits, np.max(x_i))]
        # Integers we take the values directly
        elif self.features.types[i] in ["ordinal", "num_int"]:
            splits = np.sort(np.unique(x_i))[:-1]
        elif self.features.types[i] == "bool":
            x_i = np.unique(x_i)
            if len(x_i) == 1:
                splits = []
            else:
                splits = np.array([0])
        else:
            raise Exception("Nominal features are not yet supported")

        return splits
    

    def get_best_split(self, impurity, curr_node, instances_idx):
        """ Return the optimal split along a feature """
        best_feature_split = 0
        best_obj = impurity
        best_obj_left = 0
        best_obj_right = 0
        best_split = None
        for feature in range(self.D):
            splits, N_left, N_right, objective_left, objective_right = \
                                            self.get_split(instances_idx, feature)
            # No split was conducted
            if len(splits) == 0:
                if self.save_losses:
                    curr_node.splits.append([])
                    curr_node.objectives.append([])
            else:
                
                # Otherwise search for the best split
                objective = (objective_right+objective_left) / len(instances_idx)
                if self.save_losses:
                    curr_node.splits.append(splits)
                    curr_node.objectives.append(objective)
                
                best_split_idx = np.argmin(objective)
                if objective[best_split_idx] < best_obj:
                    best_split = splits[best_split_idx]
                    best_obj = objective[best_split_idx]
                    best_obj_left = objective_left[best_split_idx] / N_left[best_split_idx]
                    best_obj_right = objective_right[best_split_idx] / N_right[best_split_idx]
                    best_feature_split = feature
    
        return best_feature_split, best_split, best_obj, best_obj_left, best_obj_right
    

    @abstractmethod
    def get_split(self, instances_idx, feature):
        """ Get the objective value at each split """
        pass
    

    @abstractmethod
    def fit(self, X, **kwargs):
        """ Fit the tree via the provided tensors """
        pass

    
    def _tree_builder(self, instances_idx, parent, depth, impurity):
        
        # Create a node
        curr_node = Node(instances_idx, parent, depth, impurity*self.impurity_factor)

        # Stop the tree growth
        if curr_node.depth >= self.max_depth or \
            impurity * self.impurity_factor < self.negligible_impurity:
            # Create a leaf
            curr_node.group = self.n_groups
            self.n_groups += 1
            data_ratio = len(instances_idx) / self.N
            self.total_impurity += impurity * data_ratio * self.impurity_factor
            return curr_node
        
        # Otherwise Find best split
        best_feature_split, best_split, best_obj, best_obj_left, best_obj_right = \
                                    self.get_best_split(impurity, curr_node, instances_idx)

        # Stop the tree growth if the decrease in Loss 
        # induced by the split is minimal or no split was conducted
        if best_obj > self.relative_decrease * impurity or best_split is None:
            # Create a leaf
            curr_node.group = self.n_groups
            self.n_groups += 1
            data_ratio = len(instances_idx) / self.N
            self.total_impurity += impurity * data_ratio * self.impurity_factor
            return curr_node
        
        # Select instances of the chosen feature
        x_i = self.X[instances_idx, best_feature_split]

        # Update the node with feature and threshold used
        curr_node.update(best_feature_split, best_split)

        # Go left
        curr_node.child_left = self._tree_builder(instances_idx[x_i <= best_split],
                                                  parent=curr_node, 
                                                  depth=depth+1,
                                                  impurity=best_obj_left)
        # Go right
        curr_node.child_right= self._tree_builder(instances_idx[x_i > best_split],
                                                  parent=curr_node, 
                                                  depth=depth+1,
                                                  impurity=best_obj_right)

        return curr_node


    def predict(self, X_new, latex_rules=False):
        """ Return the group index and the corresponding rules """
        groups = np.zeros(X_new.shape[0], dtype=np.int32)
        rules = {}
        curr_rule = []
        self._tree_traversal(self.root, np.arange(X_new.shape[0]), X_new, groups, 
                             rules, curr_rule, latex_rules)
        return groups, rules


    def postprocess_rules(self, curr_rule, latex_rules):
        """ 
        Simplify numerical rules
        - Remove redundancy x1>3 and x1>5 becomes x1>5
        - Intervals x1>3 and x1<5 becomes 3<x1<5
        """
        
        curr_rule_copy = deepcopy(curr_rule)
        separators = [SYMBOLS[latex_rules]["leq"], SYMBOLS[latex_rules]["up"]]
        select_rules_0 = [rule for rule in curr_rule_copy if separators[0] in rule]
        splits_0 = [rule.split(separators[0])+[0] for rule in select_rules_0]
        select_rules_1 = [rule for rule in curr_rule_copy if separators[1] in rule]
        splits_1 = [rule.split(separators[1])+[1] for rule in select_rules_1]
        # There are splits
        if splits_0 or splits_1:
            splits = np.array(splits_0 + splits_1)
            select_features, inv, counts = np.unique(splits[:, 0], 
                                        return_inverse=True, return_counts=True)
            # There is redundancy
            if np.any(counts >= 2):
                # Iterate over redundant features
                for i in np.where(counts >= 2)[0]:
                    select_feature = select_features[i]
                    idxs = np.where(inv == i)[0]
                    # Remove the redundant rules
                    for idx in idxs:
                        curr_rule_copy.remove(select_feature+\
                                              separators[int(splits[idx, 2])]+\
                                              splits[idx, 1])
                    # Sort the rules in ascending order of threshold
                    argsort = idxs[np.argsort(splits[idxs, 1].astype(float))]
                    thresholds = splits[argsort, 1]
                    directions = splits[argsort, 2]
                    threshold_left = None
                    threshold_right = None
                    # We go from left to right and define the rule
                    for threshold, direction in zip(thresholds, directions):
                        # We stop at the first leq
                        if direction == '0':
                            threshold_left = threshold
                            break
                        if direction == '1':
                            threshold_right = threshold
                    # print the new rule
                    if threshold_right is None:
                        new_rule = select_feature+SYMBOLS[latex_rules]["leq"]+threshold_left
                    elif threshold_left is None:
                        new_rule = select_feature+SYMBOLS[latex_rules]["up"]+threshold_right
                    else:
                        new_rule = threshold_right + SYMBOLS[latex_rules]["low"] +\
                            select_feature + SYMBOLS[latex_rules]["leq"] + threshold_left
                    # Add the new rule
                    curr_rule_copy.append(new_rule)

        return curr_rule_copy


    def _tree_traversal(self, node, instances_idx, X_new, groups, 
                                    rules, curr_rule, latex_rules):
        
        if node.child_left is None:
            # Label the instances at the leaf
            groups[instances_idx] = node.group
            if len(curr_rule) > 1:
                # Simplify long rule lists if possible
                curr_rule_copy = self.postprocess_rules(curr_rule, latex_rules)
                if len(curr_rule_copy) > 1:
                    rules[node.group] = "(" + SYMBOLS[latex_rules]["and_str"].join(curr_rule_copy) + ")"
                else:
                    rules[node.group] = curr_rule_copy[0]
            else:
                rules[node.group] = curr_rule[0]
        else:
            x_i = X_new[instances_idx, node.feature]

            feature_name = self.features.names[node.feature]
            feature_type = self.features.types[node.feature]

            # Boolean
            if feature_type == "bool":
                assert np.isclose(node.threshold, 0)
                curr_rule.append(f"not {feature_name}")
            # Ordinal
            elif feature_type == "ordinal":
                categories = np.array(self.features.maps[node.feature].cats)
                cats_left = categories[:int(node.threshold)+1]
                if len(cats_left) == 1:
                    curr_rule.append(f"{feature_name}={cats_left[0]}")
                else:
                    curr_rule.append(f"{feature_name} " + SYMBOLS[latex_rules]['in_set'] \
                                     + " [" + ",".join(cats_left)+"]")
            # Numerical
            else:
                curr_rule.append(feature_name + SYMBOLS[latex_rules]['leq'] +\
                                f"{node.threshold:.2f}")
            

            # Go left
            self._tree_traversal(node.child_left, 
                                 instances_idx[x_i <= node.threshold],
                                 X_new, groups, rules, curr_rule, latex_rules)
            curr_rule.pop()


            # Boolean
            if feature_type == "bool":
                curr_rule.append(f"{feature_name}")
            # Ordinal
            elif feature_type == "ordinal":
                cats_right = categories[int(node.threshold)+1:]
                if len(cats_right) == 1:
                    curr_rule.append(f"{feature_name}={cats_right[0]}")
                else:
                    curr_rule.append(f"{feature_name} " + SYMBOLS[latex_rules]['in_set'] \
                                     + " [" + ",".join(cats_right)+"]")
            # Numerical
            else:
                curr_rule.append(feature_name + SYMBOLS[latex_rules]['up'] +\
                                f"{node.threshold:.2f}")
            
            # Go right
            self._tree_traversal(node.child_right, 
                                 instances_idx[x_i > node.threshold],
                                 X_new, groups, rules, curr_rule, latex_rules)
            curr_rule.pop()



class L2CoETree(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, X, H):
        self.X = X
        self.N, self.D = X.shape
        self.H = H
        self.f = self.H[np.arange(self.N), np.arange(self.N)]
        self.impurity_factor = 100 / self.f.var() # To have an impurity 0-100%
        self.total_impurity = 0
        self.n_groups = 0
        impurity = np.mean((self.f - self.H.mean(1))**2)
        # Start recursive tree growth
        self.root = self._tree_builder(np.arange(self.N), parent=None, 
                                       depth=0, impurity=impurity)
        return self


    def get_split(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], [], [], []
        
        # Otherwise we optimize the objective
        N_left = np.zeros(len(splits))
        N_right = np.zeros(len(splits))
        objective_left = np.zeros(len(splits))
        objective_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        f = self.f[instances_idx]
        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            N_left[i] = len(left)
            N_right[i] = len(right)
            to_keep[i] = min(N_left[i], N_right[i]) >= self.samples_leaf
            objective_left[i] = np.sum((f[x_i <= split] - self.H[left, left.T].mean(-1))**2)
            objective_right[i] = np.sum((f[x_i > split] - self.H[right, right.T].mean(-1))**2)
        
        return splits[to_keep], N_left[to_keep], N_right[to_keep],\
                    objective_left[to_keep], objective_right[to_keep]



class PFITree(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, X, H):
        self.X = X
        self.N, self.D = X.shape
        self.H = H[..., 1:]
        self.f = H[0, :, 0]
        self.impurity_factor = 100 / self.f.var() # To have an impurity 0-100%
        self.total_impurity = 0
        self.n_groups = 0
        impurity = np.mean(np.sum((self.H.mean(0) + self.H.mean(1))**2, axis=-1))
        # Start recursive tree growth
        self.root = self._tree_builder(np.arange(self.N), parent=None, 
                                       depth=0, impurity=impurity)
        return self


    def get_split(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], [], [], []
        
        # Otherwise we optimize the objective
        N_left = np.zeros(len(splits))
        N_right = np.zeros(len(splits))
        objective_left = np.zeros(len(splits))
        objective_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            N_left[i] = len(left)
            N_right[i] = len(right)
            to_keep[i] = min(N_left[i], N_right[i]) >= self.samples_leaf
            R_left = self.H[left, left.T]
            R_right = self.H[right, right.T]
            objective_left[i] = np.sum((R_left.mean(0) + R_left.mean(1))**2)
            objective_right[i] = np.sum((R_right.mean(0) + R_right.mean(1))**2)
        
        return splits[to_keep], N_left[to_keep], N_right[to_keep],\
                    objective_left[to_keep], objective_right[to_keep]


class GADGET_PDP(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def get_split(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], [], [], []
        
        # Otherwise we optimize the objective
        N_left = np.zeros(len(splits))
        N_right = np.zeros(len(splits))
        objective_left = np.zeros(len(splits))
        objective_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            N_left[i] = len(left)
            N_right[i] = len(right)
            to_keep[i] = min(N_left[i], N_right[i]) >= self.samples_leaf
            R_left = self.R[left, left.T]
            R_right = self.R[right, right.T]
            errors_left = (R_left - R_left.mean(axis=0, keepdims=True) - 
                            R_left.mean(axis=1, keepdims=True) +
                            R_left.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True))**2
            objective_left[i] = errors_left.sum(-1).mean(-1).sum()
            errors_right = (R_right - R_right.mean(axis=0, keepdims=True) - 
                            R_right.mean(axis=1, keepdims=True) +
                            R_right.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True))**2
            objective_right[i] = errors_right.sum(-1).mean(-1).sum()
        
        return splits[to_keep], N_left[to_keep], N_right[to_keep],\
                    objective_left[to_keep], objective_right[to_keep]


    def fit(self, X, H):
        self.X = X
        self.N, self.D = X.shape
        self.f = H[0, :, 0]
        H = H + self.f.reshape((1, -1, 1))
        self.R = H[..., 1:]  # (N, N, d) tensor s.t. R_ijk = h_(r_{k}(x^(j), x^(i)))
        self.impurity_factor = 100 / self.f.var() # To have an impurity 0-100%
        self.total_impurity = 0
        self.n_groups = 0
        impurity = np.mean(np.sum((self.R - self.R.mean(axis=0, keepdims=True) - 
                                   self.R.mean(axis=1, keepdims=True) +
                                   self.R.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True))**2, 
                                   axis=-1))
        # Start recursive tree growth
        self.root = self._tree_builder(np.arange(self.N), parent=None, 
                                       depth=0, impurity=impurity)
        return self
    


class CART(FDTree):
    """ 
    Classic CART that minimizes the Squared error
    sum_leaf sum_{x^(i)\in leaf} ( h(x^(i)) - v_leaf ) ^ 2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, X, f):
        self.X = X
        self.N, self.D = X.shape
        self.f = f
        self.impurity_factor = 100 / self.f.var() # To have an impurity 0-100%
        self.total_impurity = 0
        self.n_groups = 0
        impurity = self.f.var()
        # Start recursive tree growth
        self.root = self._tree_builder(np.arange(self.N), parent=None, 
                                       depth=0, impurity=impurity)
        return self


    def get_split(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self.get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], [], [], []
        
        # Otherwise we optimize the objective
        N_left = np.zeros(len(splits))
        N_right = np.zeros(len(splits))
        objective_left = np.zeros(len(splits))
        objective_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split]
            right = instances_idx[x_i > split]
            N_left[i] = len(left)
            N_right[i] = len(right)
            to_keep[i] = min(N_left[i], N_right[i]) >= self.samples_leaf
            objective_left[i] = np.sum((self.f[left] - self.f[left].mean())**2)
            objective_right[i] = np.sum((self.f[right] - self.f[right].mean())**2)
        
        return splits[to_keep], N_left[to_keep], N_right[to_keep],\
                    objective_left[to_keep], objective_right[to_keep]



class RandomTree(L2CoETree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def get_best_split(self, impurity, curr_node, instances_idx):
        splits = []
        while len(splits) == 0:
            best_feature_split = np.random.choice(range(self.D))
            splits, N_left, N_right, objective_left, objective_right = \
                                        self.get_split(instances_idx, best_feature_split)
        # Chose a random split
        idx = np.random.choice(range(max(1, len(splits)-1)))
        
        # Otherwise search for the best split
        best_split = splits[idx]
        best_obj = (objective_right[idx]+objective_left[idx]) / len(instances_idx)
        best_obj_left = objective_left[idx] / N_left[idx]
        best_obj_right = objective_right[idx] / N_right[idx]
    
        return best_feature_split, best_split, best_obj, best_obj_left, best_obj_right


# TODO GADGET SHAP



@dataclass
class Partition:
    type: str = "l2coe"  # Type of partitionning "fd-tree" "random"
    save_losses : bool = True, # Save the tree locally
    negligible_impurity : float = 1 # When is the impurity considered low
    relative_decrease : float = 0.9 # Split is considered if the impurity decreases by AT LEAST this ratio
    samples_leaf : int = 20 # Minimum number of samples per leaf of the Fd-Tree

PARTITION_CLASSES = {
    "l2coe": L2CoETree,
    "pfi": PFITree,
    "gadget-pdp" : GADGET_PDP,
    "cart" : CART,
    "random" : RandomTree,
}



from sklearn.tree import DecisionTreeRegressor


class PartitionPartialDependence(object):

    def __init__(self, h, X, feature_names):
        self.h = h
        self.X = X
        self.N, self.d = X.shape
        self.feature_names = [feature_names.copy() for _ in range(self.d)]
        for i in range(self.d):
            self.feature_names[i].pop(i)
        self.partial_dep = [0 for _ in range(self.d)]
        self.interactions = [0 for _ in range(self.d)]
        self.tree = [0 for _ in range(self.d)]
        self.error_baseline = [0 for _ in range(self.d)]
    

    def partial_dependence(self, feature, n_steps=10):
        if not self.partial_dep[feature] == 0:
            print("Partial Dependence has already been computed")
        else:
            line = np.linspace(self.X[:, feature].min(),
                               self.X[:, feature].max(),
                               n_steps)
            self.partial_dep[feature] = np.zeros((self.N, n_steps))
            for i, target in enumerate(line):
                X_partial = self.X.copy()
                X_partial[:, feature] = target
                self.partial_dep[feature][:, i] = self.h(X_partial)
            return line


    def fit_tree(self, feature, **kwargs):
        # Create a multivariate regression dataset
        y = np.diff(self.partial_dep[feature], axis=1)
        error_baseline = np.mean(y.var(0))
        if error_baseline <= 0.001:
            print(f"H1 loss {error_baseline:.4f}")
            print("No Interactions so we do not fit a tree")
        else:
            if "min_impurity_decrease" in kwargs:
                error_baseline = np.mean(y.var(0))
                kwargs["min_impurity_decrease"] *= error_baseline
            # We regress on all other features
            features = [i for i in range(self.d) if not i == feature]
            self.tree[feature] = DecisionTreeRegressor(**kwargs).fit(self.X[:, features], y)
            self.interactions[feature] = self.tree[feature].feature_importances_


    def print_tree(self, feature, verbose=False):
        self.curr_group_idx = 0
        self.curr_feature = feature
        if self.tree[feature] == 0:
            raise Exception(f"No tree has been fitted for this feature {feature}")
        else:
            self.recurse_print_tree(self.tree[feature], verbose=verbose)
    

    def recurse_print_tree(self, tree, node=0, depth=0, verbose=False):
        # Leaf
        if tree.tree_.children_left[node] < 0:
            if verbose:
                print("|   " * depth + f"H1 loss {tree.tree_.impurity[node]:.4f}")
                print("|   " * depth + f"Samples {tree.tree_.n_node_samples[node]:d}")
            print("|   " * depth + f"Group {self.curr_group_idx}")
            self.curr_group_idx += 1
        # Internal node
        else:
            if verbose:
                print("|   " * depth + f"H1 loss {tree.tree_.impurity[node]:.4f}")
                print("|   " * depth + f"Samples {tree.tree_.n_node_samples[node]:d}")
            node_feature_name = self.feature_names[self.curr_feature][tree.tree_.feature[node]]
            print("|   " * depth + 
                f"If {node_feature_name} <= {tree.tree_.threshold[node]:.4f}:")
            self.recurse_print_tree(tree, node=tree.tree_.children_left[node], 
                                    depth=depth+1, verbose=verbose)
            print("|   " * depth + "else:")
            self.recurse_print_tree(tree, node=tree.tree_.children_right[node], 
                                    depth=depth+1, verbose=verbose)


    def get_partition_idx(self, feature):
        if self.tree[feature] == 0:
            raise Exception(f"No tree has been fitted for this feature {feature}")
        # Partition via the leaf index
        features = [i for i in range(self.d) if not i == feature]
        partition_idx = self.tree[feature].apply(self.X[:, features])
        return partition_idx
