import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


data_path = 'gs://mgb-loremipsum-dev-data/olivia/'

train_df = pd.read_csv(data_path +'train.csv')
test_df = pd.read_csv(data_path +'test.csv')
valid_df = pd.read_csv(data_path + 'valid.csv')


def exploratory_analysis(train_df, test_df, valid_df):
    tfidf_vectorizer=TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words='english')
    X_train_text = tfidf_vectorizer.fit_transform(train_df['text'])
    X_test_text = tfidf_vectorizer.transform(test_df['text'])
    X_val_text = tfidf_vectorizer.transform(valid_df['text'])



    logit = LogisticRegression(C=5e1, solver='lbfgs', random_state=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    cv_results = cross_val_score(logit, X_train_text, train_df['label'], cv=skf)
    logit.fit(X_train_text, train_df['label'])
    val_preds_log = logit.predict(X_val_text)
    confusion_matrix_logistic = confusion_matrix(valid_df['label'], val_preds_log)
    f1_score_log = f1_score(valid_df['label'], val_preds_log)

    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_text, train_df['label'])
    val_preds_nb = naive_bayes_classifier.predict(X_val_text)
    confusion_matrix_nb = confusion_matrix(valid_df['label'], val_preds_nb)
    f1_score_nb = f1_score(valid_df['label'], val_preds_nb)

    decision_tree_classifier = DecisionTreeClassifier(random_state=0)
    decision_tree_classifier.fit(X_train_text, train_df['label'])
    val_preds_dt = decision_tree_classifier.predict(X_val_text)
    confusion_matrix_dt = confusion_matrix(valid_df['label'], val_preds_dt)
    f1_score_dt = f1_score(valid_df['label'], val_preds_dt)

    lsvc = svm.LinearSVC()
    lsvc.fit(X_train_text, train_df['label'])
    val_preds_lsvc = lsvc.predict(X_val_text)
    confusion_matrix_lsvc = confusion_matrix(valid_df['label'], val_preds_lsvc)
    f1_score_lsvc = f1_score(valid_df['label'], val_preds_lsvc)

    prediction = lsvc.predict(X_test_text)
