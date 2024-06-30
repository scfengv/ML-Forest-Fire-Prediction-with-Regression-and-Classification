import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from patsy import dmatrices
from sklearn import svm, metrics
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from statsmodels.sandbox.regression.predstd import wls_prediction_std

warnings.filterwarnings("ignore")

class FireDataProcessor:
    def __init__(self, file_path, cat_col, con_col) -> None:
        self.file_path = file_path
        self.cat_col = cat_col
        self.con_col = con_col
        self.df = None
        self.x = None
        self.y = None
        self.processed_data = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def preprocess_data(self):
        self.x = self.df.drop(columns = ['area_boxcox'])
        self.y = self.df['area_boxcox']

        cat_dummy = pd.get_dummies(self.x, columns = self.cat_col, dtype = float)
        d = pd.DataFrame(cat_dummy)
        con = d[self.con_col]
        cat = d.drop(columns = self.con_col)

        # Standardize Continuous data
        scl = StandardScaler()
        con_scl = scl.fit_transform(con)

        # Data with Standardized continuous data + Dummy categorical data
        self.processed_data = np.hstack((con_scl, cat))
        col = d.columns
        
        return self.processed_data, self.y, col
    

class FireModel:
    def __init__(self, x_train, y_train, x_test, y_test, col) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.col = col
    
    def train_and_evaluate(self, model):
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)

        try:  ## SVR do not support
            coef = pd.Series(model.coef_, self.col).sort_values()
            fig, ax = plt.subplots(figsize = (10, 6))
            coef.plot(kind = 'bar', ax = ax)
            plt.title(f"{model.__class__.__name__} Features Coefficient")
            plt.xticks(rotation = 45)
            plt.savefig(f"result/{model.__class__.__name__} Features Coefficient", dpi = 300)
            plt.show()
        except:
            pass

        ## Inverse Box-Cox
        y_pred_inv = []
        y_test_inv = []
        boxcox_param = -0.06156404347097244

        for y in y_pred:
            y_pred_inv.append(inv_boxcox(y, boxcox_param))

        for y_ in self.y_test.values:
            y_test_inv.append(inv_boxcox(y_, boxcox_param))

        mse = metrics.mean_squared_error(y_true = y_test_inv, y_pred = y_pred_inv)
        print(f"{model.__class__.__name__} mse: {np.round(mse, 2)}")

        r2 = metrics.r2_score(y_true = y_test_inv, y_pred = y_pred_inv)
        print(f"{model.__class__.__name__} R2: {np.round(r2, 2)}")

        sns.regplot(x = y_pred_inv, y = y_test_inv, ci = False, line_kws = {'color': 'red'})
        plt.title(f"{model.__class__.__name__} result")
        plt.xlabel(f"Predict")
        plt.ylabel(f"True")
        plt.savefig(f"result/{model.__class__.__name__} result", dpi = 300)
        plt.show()

def main():
    os.chdir("/Users/shenchingfeng/GitHub/ML-Forest-Fire-Prediction-with-Regression-and-Classification")
    file_path = 'data/transformed_df_n0.csv'
    cat_col = ['X', 'Y', 'month', 'day']
    con_col = ['rain', 'FFMC_boxcox', 'DMC_boxcox', 'DC_boxcox', 'ISI_boxcox', 'temp_boxcox', 'RH_boxcox', 'wind_boxcox']

    # Process data
    processor = FireDataProcessor(file_path, cat_col, con_col)
    processor.load_data()
    x, y, col = processor.preprocess_data()

    # Train and evaluate models
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
    fire_model = FireModel(x_train, y_train, x_test, y_test, col)

    # Linear Regression
    linear_regression = LinearRegression()
    fire_model.train_and_evaluate(linear_regression)

    # Ridge Regression
    ridge_regression = Ridge()
    fire_model.train_and_evaluate(ridge_regression)

    # SVM
    svr = svm.SVR()
    fire_model.train_and_evaluate(svr)

if __name__ == '__main__':
    main()