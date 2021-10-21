from sklearn.tree import DecisionTreeClassifier
import numpy as np

class Boosting:
    def __init__(self, dataset, T) -> None:
        self.dataset = dataset
        self.T = T
        self.alphas = None
        self.models = None
        self.accuracy = []

    def train_decision_tree_stump_with_sample_weight(self, X, Y, weights):
        ''' train decision trees ''' 
        decision_tree_model = DecisionTreeClassifier(criterion='gini', max_depth=1)
        decision_tree_fitted = decision_tree_model.fit(X,Y, sample_weight=weights)
        return decision_tree_fitted
    
    def calculate_misclassification_rate(self, predictions, Y_pred):
        ''' calculate the misclassification rate for prediction of decision tree stump'''
        misclassified = predictions[predictions!=Y_pred]
        misclassification_rate = len(misclassified)/len(predictions)
        return misclassification_rate

    def calculate_error_rate(self, predictions, Y_pred, weights):
        ''' calculate the weighted misclassification rate'''
        misclassified_indicator = np.where((predictions!=Y_pred), 1,0)
        weights_misclassified = weights*misclassified_indicator
        error = np.sum(weights_misclassified)/np.sum(weights)
        return error
    
    def calculate_alpha(self, error) -> float:
        ''' calculate alpha value for updating the weights'''
        alpha = np.log((1-error)/error)
        return alpha

    def update_weights_with_alpha(self, alpha, weights, misclassified_indicator):
        weights *= np.exp(misclassified_indicator*alpha)
        return weights

    def run_boosting_algorithm(self, T, dataset):
        ''' Build the decision tree stamps sequentially T times and assign the updated weights'''
        alphas=[]
        models=[]
        accuracy=[]
        X = dataset[:, :-1]
        Y = dataset[:, -1]
        sample_size = X.shape[0]
        weights = np.ones(sample_size)/sample_size
        for t in range(T):
            decision_tree_fitted = self.train_decision_tree_stump_with_sample_weight(X,Y,weights)
            models.append(decision_tree_fitted)
            Y_pred = decision_tree_fitted.predict(X)
            misclassified_indicator = np.where((Y!=Y_pred), 1,0)
            misclassification_rate = self.calculate_misclassification_rate(Y, Y_pred)
            accuracy.append(misclassification_rate)
            error = self.calculate_error_rate(Y, Y_pred, weights)
            alpha = self.calculate_alpha(error)
            alphas.append(alpha)
            weights = self.update_weights_with_alpha(alpha, weights, misclassified_indicator)
        self.alphas = alphas
        self.models = models
        self.accuracy = accuracy

    def fit(self, X, Y):
        dataset = np.concatenate((X,Y), axis=1)
        self.run_boosting_algorithm(self.T, dataset)

    def predict(self, X):
        predictions = []
        for alpha, model in zip(self.alphas, self.models):
            weighted_prediction = alpha*model.predict(X)
            predictions.append(weighted_prediction)
        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))

  
