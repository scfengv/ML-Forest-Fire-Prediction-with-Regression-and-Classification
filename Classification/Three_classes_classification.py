import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import classification_report, RocCurveDisplay

warnings.filterwarnings("ignore")

class FireDataProcessor:
    def __init__(self, file_path, cat_col, con_col):
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
        target = ["no fire", "small fire", "large fire"]

        for i in range(len(self.df)):

            if self.df['area'][i] == 0:
                self.df['area'][i] = "no fire"

            elif 0 < self.df['area'][i] < 6.37:
                self.df['area'][i] = "small fire"

            elif self.df['area'][i] >= 6.37:
                self.df['area'][i] = "large fire"

        self.x = self.df.drop(columns = ['area'])
        self.y = self.df['area']

        cat_dummy = pd.get_dummies(self.x, columns = self.cat_col, dtype = float)
        d = pd.DataFrame(cat_dummy)
        con = d[self.con_col]
        cat = d.drop(columns = self.con_col)

        # Standardize Continuous data
        scl = StandardScaler()
        con_scl = scl.fit_transform(con)

        # Data with Standardized continuous data + Dummy categorical data
        self.processed_data = np.hstack((con_scl, cat))

        return self.processed_data, self.y

    def apply_smote(self):
        smote = SMOTENC(categorical_features = [self.cat_col.index(col) for col in self.cat_col], sampling_strategy = "auto")
        x_smote, y_smote = smote.fit_resample(self.x, self.y)

        cat_dummy = pd.get_dummies(x_smote, columns = self.cat_col, dtype = float)
        d = pd.DataFrame(cat_dummy)
        con = d[self.con_col]
        cat = d.drop(columns = self.con_col)

        scl = StandardScaler()
        con_scl = scl.fit_transform(con)

        return np.hstack((con_scl, cat)), y_smote

class FireModel:
    def __init__(self, x_train, y_train, x_test, y_test, target_names):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.target_names = target_names

    def train_and_evaluate(self, model):
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        y_pred_score = model.predict_proba(self.x_test)
        
        print(classification_report(self.y_test, y_pred, target_names = self.target_names))
        
        label_binarizer = LabelBinarizer().fit(self.y_train)
        y_test_bin = label_binarizer.transform(self.y_test)

        fig, ax = plt.subplots()

        for id in range(3):
            RocCurveDisplay.from_predictions(
                y_test_bin[:, id],
                y_pred_score[:, id],
                name = f"ROC Curve for '{model.classes_[id]}'",
                ax = ax
            )
        
        plt.plot([0, 1], [0, 1], color = 'black', linestyle = '--')
        plt.title(f"{model.__class__.__name__}\nOne vs Rest ROC curve")
        plt.savefig(f"result/Three_classes_{model.__class__.__name__}\nOne vs Rest ROC curve", dpi = 300)
        plt.show()

def main():
    os.chdir("/Users/shenchingfeng/GitHub/ML-Forest-Fire-Prediction-with-Regression-and-Classification")
    file_path = 'data/transformed_df.csv'
    cat_col = ['X', 'Y', 'month', 'day']
    con_col = ['rain_YJ', 'FFMC_YJ', 'DMC_YJ', 'DC_YJ', 'ISI_YJ', 'temp_YJ', 'RH_YJ', 'wind_YJ']
    target_names = ["no fire", "small fire", "large fire"]

    # Process data
    processor = FireDataProcessor(file_path, cat_col, con_col)
    processor.load_data()
    x, y = processor.preprocess_data()

    # Train and evaluate models without SMOTE
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
    fire_model = FireModel(x_train, y_train, x_test, y_test, target_names)

    # Logistic Regression
    logistic_regression_model = LogisticRegression()
    fire_model.train_and_evaluate(logistic_regression_model)

    # SVM
    svm_model = svm.SVC(probability = True)
    fire_model.train_and_evaluate(svm_model)

    # Apply SMOTE and re-evaluate
    x_smote, y_smote = processor.apply_smote()
    x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, train_size=0.8)
    fire_model = FireModel(x_train, y_train, x_test, y_test, target_names)

    # Logistic Regression with SMOTE
    fire_model.train_and_evaluate(logistic_regression_model)

    # SVM with SMOTE
    fire_model.train_and_evaluate(svm_model)

if __name__ == "__main__":
    main()