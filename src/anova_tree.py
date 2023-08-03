
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




class ANOVATree(BaseEstimator):
    def __init__(self, feature_names, max_depth=3, negligible_impurity=1e-5, 
                 relative_decrease=0.7, save_losses=False):
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.negligible_impurity = negligible_impurity
        self.relative_decrease = relative_decrease
        self.save_losses = save_losses


    def print(self, verbose=False):
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
            curr_feature_name = self.feature_names[node.feature]
            print("|   " * node.depth + f"If {curr_feature_name} <= {node.threshold:.4f}:")
            self.recurse_print_tree(node=node.child_left, verbose=verbose)
            print("|   " * node.depth + "else:")
            self.recurse_print_tree(node=node.child_right, verbose=verbose)


    def get_split(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]
        if len(x_i) < 50:
            splits = np.quantile(x_i, [0.25, 0.5, 0.75])
        else:
            splits = np.quantile(x_i, np.arange(1, 10) / 10)
        # splits = np.linspace(-1, 1, 11)
        # splits = [-0.75, -0.25, 0, 0.25, 0.75]
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
        # Set the Anchored decomposition
        self.A = A
        self.f = self.A[np.arange(self.N), np.arange(self.N)]
        self.n_groups = 0
        # Compute the ANOVA 1 matrix (NxNx(d+1))
        # Start recursive tree growth
        impurity = np.mean((self.f - self.A.mean(1))**2)
        self.root = self._tree_builder(np.arange(self.N), parent=None, 
                                       depth=0, impurity=impurity)
        return self
    

    def predict(self, X_new):
        groups = np.zeros(X_new.shape[0], dtype=np.int)
        self._tree_traversal(self.root, np.arange(X_new.shape[0]), X_new, groups)
        return groups


    def _tree_traversal(self, node, instances_idx, X_new, groups):
        
        if node.child_left is None:
            # Label the instances at the leaf
            groups[instances_idx] = node.group
        else:
            x_i = X_new[instances_idx, node.feature]

            # Go left
            self._tree_traversal(node.child_left, 
                                 instances_idx[x_i <= node.threshold],
                                 X_new, groups)
            # Go right
            self._tree_traversal(node.child_right, 
                                 instances_idx[x_i > node.threshold],
                                 X_new, groups)

    # def _nodes(self, tree, lvl=0):
    #     """
    #     Enumerates all the nodes in the tree

    #     Parameters
    #     ----------
    #     tree: Thing
    #         Tree node
    #     lvl: int (default 0)
    #         Tree level

    #     Yields
    #     ------
    #     Thing:
    #         Current child node
    #     int:
    #         Level of current child node

    #     Note
    #     ----
    #     + Thing is a generic container, in this case its a node in the tree.
    #     + You'll find it in <src.tools.containers>
    #     """

    #     if tree:
    #         yield tree, lvl
    #         for kid in tree.kids:
    #             lvl1 = lvl
    #             for sub, lvl1 in self._nodes(kid, lvl1 + 1):
    #                 yield sub, lvl1

    # @staticmethod
    # def _path_from_root(node):
    #     """
    #     All the attributes in the path from root to node

    #     Parameters
    #     ----------
    #     node : Thing
    #         The tree node object

    #     Returns
    #     -------
    #     list:
    #         A list of all the attributes from root to node
    #     """

    #     path_names = [keys for keys in map(lambda x: x[0], node.branch)]
    #     return path_names

    # def _leaves(self, thresh=float("inf")):
    #     """
    #     Enumerate all leaf nodes

    #     Parameters
    #     ----------
    #     thresh: float (optional)
    #         When provided. Only leaves with values less than thresh are returned

    #     Yields
    #     ------
    #     Thing:
    #         Leaf node

    #     Note
    #     ----
    #     + Thing is a generic container, in this case its a node in the tree.
    #     + You'll find it in <src.tools.containers>
    #     """

    #     for node, _ in self._nodes(self.tree):
    #         if not node.kids and node.score <= thresh:
    #             yield node

    # def _find(self, test_instance, tree_node=None):
    #     """
    #     Find the leaf node that a given row falls in.

    #     Parameters
    #     ----------
    #     test_instance: <pandas.frame.Series>
    #         Test instance

    #     Returns
    #     -------
    #     Thing:
    #         Node where the test instance falls

    #     Note
    #     ----
    #     + Thing is a generic container, in this case its a node in the tree.
    #     + You'll find it in <src.tools.containers>
    #     """

    #     if len(tree_node.kids) == 0:
    #         found = tree_node
    #     else:
    #         for kid in tree_node.kids:
    #             found = kid
    #             if kid.val[0] <= test_instance[kid.f] < kid.val[1]:
    #                 found = self._find(test_instance, kid)
    #             elif kid.val[1] == test_instance[kid.f] \
    #                             == self.tree.t.describe()[kid.f]['max']:
    #                 found = self._find(test_instance, kid)

    #     return found


    # @staticmethod
    # def pairs(lst):
    #     """
    #     Return pairs of values form a list

    #     Parameters
    #     ----------
    #     lst: list
    #         A list of values

    #     Yields
    #     ------
    #     tuple:
    #         Pair of values

    #     Example
    #     -------

    #     BEGIN
    #     ..
    #     lst = [1,2,3,5]
    #     ..
    #     returns -> 1,2
    #     lst = [2,3,5]
    #     ..
    #     returns -> 2,3
    #     lst = [3,5]
    #     ..
    #     returns -> 3,5
    #     lst = []
    #     ..
    #     END
    #     """
    #     while len(lst) > 1:
    #         yield (lst.pop(0), lst[0])


    # def best_plan(self, better_nodes, item_sets):
    #     """
    #     Obtain the best plan that has the maximum jaccard index
    #     with elements in an item set.

    #     Parameters
    #     ----------
    #     better_nodes: List[Thing]
    #         A list of terminal nodes that are "better" than the node
    #         which the current test instance lands on.
    #     item_set: List[set]
    #         A list containing all the frequent itemsets.

    #     Returns
    #     -------
    #     Thing:
    #         Best leaf node

    #     Note
    #     ----
    #     + Thing is a generic container, in this case its a node in the tree.
    #     + You'll find it in <src.tools.containers>
    #     """
    #     max_intersection = float("-inf")

    #     # Sort better nodes by score
    #     better_nodes.sort(key=lambda X: X.score)

    #     # Initialize the best path
    #     best_path = better_nodes[0]

    #     # Try and find a better path, with a higher overlap with item sets
    #     for node in better_nodes:
    #         change_set = set([bb[0] for bb in node.branch])
    #         for item_set in item_sets:
    #             jaccard_index = self.jaccard_similarity_score(
    #                 item_set, change_set)
    #             if 0 < jaccard_index >= max_intersection:  # TODO: Check
    #                 best_path = node
    #                 max_intersection = jaccard_index

    #     return best_path

    # def best_plan_closest(self, better_nodes, current_node):
    #     """
    #     Obtain the best plan by picking a node from better nodes that is 
    #     closest to the current_node

    #     Parameters
    #     ----------
    #     better_nodes: List[Thing]
    #         A list of terminal nodes that are "better" than the node
    #         which the current test instance lands on.
    #     current_node : [type]
    #         The node where the current test case falls into

    #     Returns
    #     -------
    #     Thing:
    #         Best leaf node

    #     Note
    #     ----
    #     + Thing is a generic container, in this case its a node in the tree.
    #     + You'll find it in <src.tools.containers>
    #     """
    #     current_path_components = self._path_from_root(current_node)
    #     min_dist = float("inf")
    #     best_path = current_node
    #     for other_path in better_nodes:
    #         other_path_components = self._path_from_root(other_path)
    #         jaccard_index = self.jaccard_similarity_score(
    #             current_path_components, other_path_components)
    #         if jaccard_index <= min_dist:
    #             min_dist = jaccard_index
    #             best_path = other_path
    #     return best_path

    # def predict(self, X_test):
    #     """
    #     Recommend plans for a test data

    #     Parameters
    #     ----------
    #     test_df: <pandas.core.frame.DataFrame>
    #         Testing data

    #     Returns
    #     -------
    #     <pandas.core.frame.DataFrame>:
    #         Recommended changes
    #     """

    #     new = []
    #     y = X_test[X_test.columns[-1]]
    #     X = X_test[X_test.columns[1:-1]]

    #     # ----- Itemset Learning -----
    #     if self.strategy == "itemset":
    #         # -- Instantiate item set learning --
    #         isl = ItemSetLearner(bins=self.bins, support_min=self.support_min)
    #         # -- Fit the data to itemset learner --
    #         isl.fit(X, y)
    #         # -- Transform into itemsets --
    #         item_sets = isl.transform()

    #     # ----- Obtain changes -----
    #     for row_num in range(len(X_test)):
    #         if X_test.iloc[row_num]["<bug"] == 1:
    #             cur = X_test.iloc[row_num]
    #             # Find the location of the current test instance on the tree
    #             pos = self._find(cur, tree_node=self.tree)
    #             # Find all the leaf nodes on the tree that atleast alpha
    #             # times smaller that current test instance
    #             better_nodes = [leaf for leaf in self._leaves(
    #                 thresh=self.alpha * pos.score)]
    #             # TODO: Check this
    #             if better_nodes:
    #                 # - Find the path with the highest overlap with itemsets -
    #                 # -- Choose the startegy based on how we want to do it --
    #                 # ---- Use item sets ----
    #                 if self.strategy == "itemset":
    #                     best_path = self.best_plan(better_nodes, item_sets)
    #                 # ---- Find the closest ----
    #                 elif self.strategy == "closest":
    #                     best_path = self.best_plan_closest(better_nodes, pos)
    #                 else:
    #                     raise ValueError(
    #                         "Invalid argument for. Use either \"itemset\" or \"closest\" ")

    #                 for entities in best_path.branch:
    #                     cur[entities[0]] = entities[1]
    #                 new.append(cur.values.tolist())
    #         else:
    #             new.append(X_test.iloc[row_num].values.tolist())

    #     new = pd.DataFrame(new, columns=X_test.columns)
    #     return new



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
