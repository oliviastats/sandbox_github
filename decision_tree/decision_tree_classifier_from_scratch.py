import numpy as np
import pandas as pd

# sources https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb

class Node:
    def __init__(self,  feature_index=None, splitting_value=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.splitting_value = splitting_value
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=2):
        # stopping conditions
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        """
        Calculate entropy per node
        """
        classes = np.unique(y)
        total = len(y)
        entropy = np.array([])
        for cl in classes:
            entropy_value_for_class = (
                -len(y[y == cl]) / total * np.log2((y[y == cl]) / total)
            )
            np.append(entropy, entropy_value_for_class)
        return np.sum(entropy)

    def gini_index(self, y):
        """
        Calculate gini index per node
        Advantage: less computationally expensive
        """
        classes = np.unique(y)
        total = len(y)
        sum_ratios_squared = 0
        for cl in classes:
            ratio_class = len(y[y == cl]) / total
            sum_ratios_squared += ratio_class ** 2
        return 1 - sum_ratios_squared

    def information_gain(self, parent_set, left_child_set, right_child_set, method="gini"):
        """
        Calculate information gain per split
        """
        if method == "entropy":
            weight_right_child = len(right_child_set) / len(parent_set)
            weight_left_child = len(left_child_set) / len(parent_set)
            information_gain = (
                self.entropy(parent_set)
                - weight_right_child * self.entropy(right_child_set)
                - weight_left_child * self.entropy(left_child_set)
            )

        else:
            weight_right_child = len(right_child_set) / len(parent_set)
            weight_left_child = len(left_child_set) / len(parent_set)
            information_gain = (
                self.gini_index(parent_set)
                - weight_right_child * self.gini_index(right_child_set)
                - weight_left_child * self.gini_index(left_child_set)
            )
        return information_gain

    def split_data(self, dataset, feature_index, splitting_value):
        subset_left = dataset[dataset[:,feature_index] <= splitting_value]
        subset_right = dataset[dataset[:,feature_index] > splitting_value]
        return subset_left, subset_right

    def determine_optimal_split(self, dataset: np.array, feature_indices):
        X = dataset[:, :-1]
        y = dataset[:,-1]
        bestsplit = {}
        information_gain = -np.inf
        feature_indices = range(X.shape[1])
        for index in feature_indices:
            feature_values = X[:,index]
            for value in feature_values:
                subset_left, subset_right = self.split_data(dataset, index, value)
                information_gain_curr = self.information_gain(y, subset_left, subset_right)
                if information_gain_curr > information_gain:
                    information_gain = information_gain_curr
                    bestsplit['feature_index'] = index
                    bestsplit['splitting_value'] = value
                    bestsplit['subset_left'] = subset_left
                    bestsplit['subset_right'] = subset_right
                    bestsplit['info_gain'] = information_gain_curr
        return bestsplit

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        X, Y = dataset[:,:-1], dataset[:,-1]
        feature_indices = range(X.shape[1])
        if curr_depth<=self.max_depth:
            best_split = self.determine_optimal_split(dataset, feature_indices)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["subset_left"], curr_depth+1)
                # recur rightS
                right_subtree = self.build_tree(best_split["subset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split['splitting_value'], 
                            left_subtree, right_subtree, best_split["info_gain"])
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
           
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
 
    def fit(self, X, Y):
        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)
        print('finish')
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.splitting_value:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)