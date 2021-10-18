import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso



data_path_gcs = 'gs://mgb-transformers-sandbox-dev-data/olivia_test/kaggle/insurance.csv'

def get_dummies_for_categorical(df: pd.DataFrame, categorical_cols: list):
    df_categorical_encoded = pd.get_dummies(df, prefix='OHE', 
            prefix_sep='_',
            columns = categorical_cols,
            drop_first= True,
            dtype='int8' )
    return df_categorical_encoded

def get_train_test_split(test_size: float, df: pd.DataFrame):
    X = df.drop(columns='charges')
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state=0)
    return X_train, X_test, y_train, y_test

def _calculate_r_squared(y_pred: np.array, y_test: np.array):
    sum_square_error = np.sum((y_pred-y_test)**2)
    sum_square_total = np.sum((y_test - y_test.mean())**2)
    r_squared = 1- sum_square_error/sum_square_total
    return r_squared

def get_model_evaluation_metrics(y_pred: np.array):
    mean_squared_error = np.mean((y_pred-y_test)**2)
    r_squared = _calculate_r_squared(y_pred, y_test)
    r_squared_adjusted = 1-((1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1))
    return mean_squared_error, r_squared, r_squared_adjusted
    
if __name__=='__main__':
    insurance_df = pd.read_csv(data_path_gcs)
    
    # data exploration
    insurance_df.shape
    insurance_df.describe()
    insurance_df.isna().sum()
    corr = insurance_df.corr()
    sns.heatmap(corr)
    insurance_df['charges'].hist()
    np.log(insurance_df['charges']).hist()
    
    sns.violinplot(x='sex', y='charges', data=insurance_df)
    sns.violinplot(x='smoker', y='charges', data=insurance_df)    
    
    plt.figure(figsize=(14,6))
    sns.boxplot(x='children', y='charges', data=insurance_df)
    plt.title('Box plot of charges vs children')

    plt.figure(figsize=(14,6))
    sns.violinplot(x='region', y='charges', data=insurance_df)
    plt.title('Violin plot of charges vs region')

    plt.figure(figsize=(14,6))
    sns.scatterplot(x='age', y='charges', data=insurance_df, hue='smoker')
    plt.title('Scatter plot of charges vs age')

    plt.figure(figsize=(16,4))
    sns.scatterplot(x='bmi', y='charges', data=insurance_df, hue='smoker')
    plt.title('Scatter plot of charges vs bmi')
    
    # data preprocessing
    insurance_df_one_hot_encoded = get_dummies_for_categorical(insurance_df, ['sex', 'children', 'smoker', 'region'])
    X_train, X_test, y_train, y_test = get_train_test_split(0.3, insurance_df_one_hot_encoded)
    
    # train linear regression model
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)
    print(list(zip(linear_regressor.coef_, X_train.columns)))
    print(linear_regressor.intercept_)

    #model evaluation
    y_pred_lin_reg = linear_regressor.predict(X_test)
    mean_squared_error_lin_reg, r_squared_lin_reg, r_squared_adjusted_lin_reg = get_model_evaluation_metrics(y_pred_lin_reg)

    # k-fold CV (using all independent variables)
    scores_r_squared_lin_reg = cross_validate(linear_regressor, X_train, y_train, scoring='r2', cv=5, return_train_score=True)
    # test scores
    # {'fit_time': array([0.00701928, 0.00410342, 0.00360608, 0.00330853, 0.00313878]),
    # 'score_time': array([0.00257277, 0.00223231, 0.00239587, 0.00235415, 0.00194931]),
    # 'test_score': array([0.76309188, 0.68900725, 0.73167505, 0.70117543, 0.71673097]),
    # 'train_score': array([0.72467205, 0.73852294, 0.73144479, 0.73802971, 0.73538798])}

    # might be overfitting due to low training error and higher test error as well as variability in the test error
    # ridge regression
    ridge_regression = Ridge(alpha=3.0)
    ridge_regression.fit(X_train, y_train)

    # model evaluation
    y_pred_ridge = ridge_regression.predict(X_test)
    mean_squared_error_ridge, r_squared_ridge, r_squared_adjusted_ridge = get_model_evaluation_metrics(y_pred_ridge)
    
    # k-fold CV (using all independent variables)
    scores_r_squared_ridge = cross_validate(ridge_regression, X_train, y_train, scoring='r2', cv=5, return_train_score=True)
    # test scores for ridge
    # {'fit_time': array([0.00347519, 0.00247622, 0.00239515, 0.0028131 , 0.00251269]),
    # 'score_time': array([0.00324559, 0.0018363 , 0.00204635, 0.00304055, 0.0016861 ]),
    # 'test_score': array([0.76132238, 0.69077202, 0.72890791, 0.70465578, 0.71618056]),
    # 'train_score': array([0.724183  , 0.73807329, 0.7310143 , 0.7375275 , 0.73481519])}

    # might be overfitting due to low training error and higher test error as well as variability in the test error
    # lasso regression
    lasso_regression = Lasso(alpha=3)
    lasso_regression.fit(X_train, y_train)

    y_pred_lasso = lasso_regression.predict(X_test)
    mean_squared_error_lasso, r_squared_lasso, r_squared_adjusted_lasso = get_model_evaluation_metrics(y_pred_lasso)
    
    # k-fold CV (using all independent variables)
    scores_r_squared_lasso = cross_validate(lasso_regression, X_train, y_train, scoring='r2', cv=5, return_train_score=True)
    # test scores for ridge
    # {'fit_time': array([0.00772047, 0.005481  , 0.00560498, 0.00501084, 0.00455189]),
    #'score_time': array([0.00325179, 0.00378561, 0.00371909, 0.00295734, 0.00301218]),
    #'test_score': array([0.76328892, 0.68949479, 0.73111222, 0.70159493, 0.71685883]),
    #'train_score': array([0.72465454, 0.73851052, 0.73142722, 0.73801238, 0.73536932])}

    
    # not much of a difference in model performance when comparing linear reg, ridge and lasso
    # decising on the linear regression model as the final model with r_squared of 0.79 
    # and r_squared adjusted of 0.78

