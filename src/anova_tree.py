
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator



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




class FDTree(BaseEstimator):
    def __init__(self, features, max_depth=3, negligible_impurity=1e-5, 
                 relative_decrease=0.7, save_losses=False):
        self.features = features
        self.max_depth = max_depth
        self.negligible_impurity = negligible_impurity
        self.relative_decrease = relative_decrease
        self.save_losses = save_losses


    def print(self, verbose=False, latex=False):
        if latex:
            self.recurse_print_tree_latex(self.root, verbose=verbose)
        else:
            self.recurse_print_tree(self.root, verbose=verbose)
    

    def recurse_print_tree(self, node, verbose=False):
        if verbose:
            print("|   " * node.depth + f"L2 Exclusion {node.impurity:.4f}")
            print("|   " * node.depth + f"Samples {len(node.instances_idx):d}")
        # Leaf
        if node.child_left is None:
            print("|   " * node.depth + f"Group {node.group}")
        # Internal node
        else:
            curr_feature_name = self.features.names[node.feature]
            print("|   " * node.depth + f"If {curr_feature_name} <= {node.threshold:.4f}:")
            self.recurse_print_tree(node=node.child_left, verbose=verbose)
            print("|   " * node.depth + "else:")
            self.recurse_print_tree(node=node.child_right, verbose=verbose)

    
    def recurse_print_tree_latex(self, node, verbose=False):
        if verbose:
            print("|   " * node.depth + f"L2 Exclusion {node.impurity:.4f}")
            print("|   " * node.depth + f"Samples {len(node.instances_idx):d}")
        # Leaf
        if node.child_left is None:
            print("|   " * node.depth + f"Group {node.group}")
        # Internal node
        else:
            curr_feature_name = self.features.names[node.feature]
            print("|   " * node.depth + f"If {curr_feature_name} <= {node.threshold:.4f}:")
            self.recurse_print_tree(node=node.child_left, verbose=verbose)
            print("|   " * node.depth + "else:")
            self.recurse_print_tree(node=node.child_right, verbose=verbose)


    def get_split_candidates(self, x_i, i):
        # Numerical features we take quantiles
        if self.features.types[i] == "num":
            if len(x_i) < 50:
                splits = np.quantile(x_i, [0.25, 0.5, 0.75])
            else:
                splits = np.quantile(x_i, np.arange(1, 10) / 10)
        # Integers we take the values directly
        elif self.features.types[i] == "num_int":
            splits = np.sort(np.unique(x_i))
        elif self.features.types[i] == "bool":
            x_i = np.unique(x_i)
            if len(x_i) == 1:
                splits = []
            else:
                splits = [0]
        else:
            raise Exception("Categorical Feature are not yet supported !!!")

        return splits
    

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
        f = self.f[instances_idx]
        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split].reshape((-1, 1))
            right = instances_idx[x_i > split].reshape((-1, 1))
            N_left[i] = len(left)
            N_right[i] = len(right)
            objective_left[i] = np.sum((f[x_i <= split] - self.A[left, left.T].mean(-1))**2)
            objective_right[i] = np.sum((f[x_i > split] - self.A[right, right.T].mean(-1))**2)
        
        return splits, N_left, N_right, objective_left, objective_right
    

    def _tree_builder(self, instances_idx, parent, depth, impurity):
        
        # Create a node
        curr_node = Node(instances_idx, parent, depth, impurity)

        # Stop the tree growth
        if curr_node.depth >= self.max_depth or impurity < self.negligible_impurity:
            # Create a leaf
            curr_node.group = self.n_groups
            self.n_groups += 1
            return curr_node
        
        # Otherwise Find best split
        best_feature_split = 0
        best_obj = impurity
        best_obj_left = 0
        best_obj_right = 0
        for feature in range(self.D):
            splits, N_left, N_right, objective_left, objective_right = \
                                            self.get_split(instances_idx, feature)
            # No split was conducted
            if len(splits) == 0:
                if self.save_losses:
                    curr_node.splits.append([])
                    curr_node.objectives.append([])
            
            # Otherwise search for the best split
            objective = (objective_right+objective_left) / (len(instances_idx))
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
        x_i = self.X[instances_idx, best_feature_split]

        # Stop the tree growth if the decrease in L2 Exclusion Cost 
        # induced by the split is minimal
        if best_obj > self.relative_decrease * impurity:
            # Create a leaf
            curr_node.group = self.n_groups
            self.n_groups += 1
            return curr_node
        
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


    def fit(self, X, A):
        self.X = X
        self.N, self.D = X.shape
        self.A = A
        self.f = self.A[np.arange(self.N), np.arange(self.N)]
        self.n_groups = 0
        impurity = np.mean((self.f - self.A.mean(1))**2)
        # Start recursive tree growth
        self.root = self._tree_builder(np.arange(self.N), parent=None, 
                                       depth=0, impurity=impurity)
        return self
    

    def predict(self, X_new):
        groups = np.zeros(X_new.shape[0], dtype=np.int)
        rules = {}
        curr_rule = []
        self._tree_traversal(self.root, np.arange(X_new.shape[0]), X_new, groups, rules, curr_rule)
        return groups, rules


    def _tree_traversal(self, node, instances_idx, X_new, groups, rules, curr_rule):
        
        if node.child_left is None:
            # Label the instances at the leaf
            groups[instances_idx] = node.group
            rules[node.group] = " & ".join(curr_rule)
        else:
            x_i = X_new[instances_idx, node.feature]

            curr_rule.append(f"{self.features.names[node.feature]} <= {node.threshold:.2f}")
            # Go left
            self._tree_traversal(node.child_left, 
                                 instances_idx[x_i <= node.threshold],
                                 X_new, groups, rules, curr_rule)
            curr_rule.pop()

            curr_rule.append(f"{self.features.names[node.feature]} > {node.threshold:.2f}")
            # Go right
            self._tree_traversal(node.child_right, 
                                 instances_idx[x_i > node.threshold],
                                 X_new, groups, rules, curr_rule)
            curr_rule.pop()



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
