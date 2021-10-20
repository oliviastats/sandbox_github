import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, max_depth=2):
        # stopping conditions
        self.max_depth = max_depth

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

    def determine_optimal_split(self, dataset: np.array):
        X = dataset[:, :-1]
        y = dataset[:,-1]
        number_features = X.shape[1]
        bestsplit = {}
        information_gain = -np.inf
        for i in range(number_features):
            feature_values = X[:,i]
            for value in feature_values:
                subset_left, subset_right = self.split_data(dataset, i, value)
                information_gain_curr = self.information_gain(y, subset_left, subset_right)
                if information_gain_curr > information_gain:
                    information_gain = information_gain_curr
                    bestsplit['feature_index'] = i
                    bestsplit['feature_value'] = value
                    bestsplit['subset_left'] = subset_left
                    bestsplit['subset_right'] = subset_right
                    bestsplit['information_gain'] = information_gain_curr
        return bestsplit
        


