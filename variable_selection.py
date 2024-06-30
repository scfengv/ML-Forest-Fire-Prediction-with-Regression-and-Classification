import os
import mlxtend
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn import svm, metrics
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

class Variable_selection:
    def __init__(self, file_path, cat_col, con_col) -> None:
        self.file_path = file_path
        self.cat_col = cat_col
        self.con_col = con_col
        self.df = None
        self.x = None
        self.y = None
        self.processed_data = None

    # def preprocessor(self):
    #     self.df = pd.read_csv(self.file_path)
    #     x_ = self.df.drop(['area'], axis = 1)
    #     y_, _ = stats.boxcox(self.df['area'])
    #     self.x = pd.get_dummies(x_, columns = self.cat_col)
    #     self.y = pd.Series(y_)
        
    #     return self.x, self.y

    def preprocessor(self):
        self.df = pd.read_csv(self.file_path)
        x_ = self.df.drop(['area'], axis = 1)
        self.x = pd.get_dummies(x_, columns = self.cat_col)
        self.y = self.df['area']

        return self.x, self.y

    def sequential_features_selection(self, forward: bool, floating: bool, cv: int, model):
        sfs = SFS(
            cv = 5,
            estimator = model,
            forward = forward,
            floating = floating,
            k_features = len(self.x.columns),
            scoring = "r2"
        )
        sfs.fit(self.x, self.y)

        metric_dict = sfs.get_metric_dict()
        k_features_ = list(metric_dict.keys())
        avg_scores = np.array([metric_dict[k]['avg_score'] for k in k_features_])
        # std_devs = np.array([metric_dict[k]['std_dev'] for k in k_features_])
        plt.scatter(x = k_features_, y = avg_scores)
        plt.title(f"SFS_all_features_{model.__class__.__name__}")
        plt.savefig(f"result/SFS_all_features_{model.__class__.__name__}_FOR_{forward}_FLOAT_{floating}", dpi = 300)
        plt.show()


    def sfs_with_k_features(self, forward: bool, floating: bool, cv: int, model, k_features: int):
        sfs = SFS(
            cv = cv, 
            estimator = model, 
            forward = forward, 
            floating = floating, 
            k_features = k_features,
            scoring = "r2"
        )
        
        sfs.fit(self.x, self.y)
        print(f"Selected {k_features}: {sfs.k_feature_names_}")
        print(f"SFS score: {sfs.k_score_}")

def main():
    os.chdir("/Users/shenchingfeng/GitHub/ML-Forest-Fire-Prediction-with-Regression-and-Classification")
    file_path = "data/forestfires.csv"
    cat_col = ['X', 'Y', 'month', 'day']
    con_col = ['rain', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']

    variable_selection = Variable_selection(file_path, cat_col, con_col)
    variable_selection.preprocessor()

    # Linear Regression
    # linear_regression = LinearRegression()
    # variable_selection.sequential_features_selection(
    #     forward = True, floating = True, cv = 5, model = linear_regression
    # )
    # variable_selection.sfs_with_k_features(
    #     forward = True, floating = True, cv = 5, model = linear_regression, k_features = 10
    # )

    # SVM
    svr = svm.SVR()
    variable_selection.sequential_features_selection(
        forward = True, floating = True, cv = 5, model = svr
    )
    variable_selection.sfs_with_k_features(
        forward = True, floating = True, cv = 5, model = svr, k_features = 10
    )

if __name__ == '__main__':
    main()
    