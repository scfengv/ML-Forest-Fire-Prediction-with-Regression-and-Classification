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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.sandbox.regression.predstd import wls_prediction_std

warnings.filterwarnings("ignore")

os.chdir("/Users/shenchingfeng/GitHub/ML-Forest-Fire-Prediction-with-Regression-and-Classification")

f = 'data/transformed_df_n0.csv'
df = pd.read_csv(f)

cat_col = ['X', 'Y', 'month', 'day']
con_col = ['rain', 'FFMC_boxcox', 'DMC_boxcox', 'DC_boxcox', 'ISI_boxcox', 'temp_boxcox', 'RH_boxcox', 'wind_boxcox']

x = df.drop(columns = ['area_boxcox'])
y = df['area_boxcox']

cat_dummy = pd.get_dummies(x, columns = cat_col, dtype = float)
d = pd.DataFrame(cat_dummy)
con = d[con_col]
cat = d.drop(columns = con_col)

## Standardize Continuous data
scl = StandardScaler()
con_scl = scl.fit_transform(con)

## Data with Standardized continuous data + Dummy categorical data
d_scl = np.hstack((con_scl, cat))

x_train, x_test, y_train, y_test = train_test_split(d_scl, y, shuffle = True, train_size = 0.8)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
print('mse: ', np.round(mse, 2))