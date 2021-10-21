import numpy as np
import pandas as pd

# sources https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb

class Node:
    def __init__(self,  feature_index=None, splitting_value=None, left=None, right=None, variance_reduction=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.splitting_value = splitting_value
        self.left = left
        self.right = right
        self.variance_reduction = variance_reduction
        
        # for leaf node
        self.value = value
class DecisionTreeRegressor:
    def __init__(self, max_depth=2):
        # stopping conditions
        self.max_depth = max_depth
        self.root = None

    def variance(self, y):
        """
        Calculate variance per node
        """
        mean = np.mean(y)
        variance = np.mean((y-mean)**2)
        return variance

    def variance_reduction(self, parent_set, left_child_set, right_child_set):
        """
        Calculate the reduction in variance per split
        """
        weight_right_child = len(right_child_set) / len(parent_set)
        weight_left_child = len(left_child_set) / len(parent_set)
        variance_reduction = (
                self.variance(parent_set)
                - weight_right_child * self.variance(right_child_set)
                - weight_left_child * self.variance(left_child_set)
            )
        return variance_reduction

    def split_data(self, dataset, feature_index, splitting_value):
        subset_left = dataset[dataset[:,feature_index] <= splitting_value]
        subset_right = dataset[dataset[:,feature_index] > splitting_value]
        return subset_left, subset_right

    def determine_optimal_split(self, dataset: np.array, feature_indices):
        X = dataset[:, :-1]
        y = dataset[:,-1]
        bestsplit = {}
        variance_reduction = -np.inf
        feature_indices = range(X.shape[1])
        for index in feature_indices:
            feature_values = np.unique(X[:,index])
            for value in feature_values:
                subset_left, subset_right = self.split_data(dataset, index, value)
                variance_reduction_curr = self.variance_reduction(y, subset_left, subset_right)
                if variance_reduction_curr > variance_reduction:
                    variance_reduction = variance_reduction_curr
                    bestsplit['feature_index'] = index
                    bestsplit['splitting_value'] = value
                    bestsplit['subset_left'] = subset_left
                    bestsplit['subset_right'] = subset_right
                    bestsplit['variance_reduction'] = variance_reduction_curr
        return bestsplit

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        X, Y = dataset[:,:-1], dataset[:,-1]
        feature_indices = range(X.shape[1])
        if curr_depth<=self.max_depth:
            best_split = self.determine_optimal_split(dataset, feature_indices)
            # check if variance reduction is positive
            if best_split["variance_reduction"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["subset_left"], curr_depth+1)
                # recur rightS
                right_subtree = self.build_tree(best_split["subset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split['splitting_value'], 
                            left_subtree, right_subtree, best_split["variance_reduction"])
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