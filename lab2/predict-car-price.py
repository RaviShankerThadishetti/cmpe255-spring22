import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt


class CarPrice:

    def __init__(self):
        self.df = pd.read_csv(
            'data\data.csv')
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)  # returns the rms

    def linear_regression(self, X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])  # Apply a column of 1s to get bias term

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        # Formula to get the array of weight parameters
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]
# base should be an array of strings

    def prepare_X(self, input_data, base):
        df_num = input_data[base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X


def calculate() -> None:
    carprice_data = CarPrice()
    carprice_data.trim()
    carPrice_data = carprice_data.df
    np.random.seed(2)
    no_of_entries = len(carPrice_data)
    # 20 percent of the data to test data set.
    no_of_entries_test = int(0.2*no_of_entries)
    # 20 percent of the data to validation data set
    no_of_entries_val = int(0.2*no_of_entries)
    # The remaining data goes to the training data set
    no_of_entries_train = no_of_entries - \
        (no_of_entries_val + no_of_entries_test)

    indx = np.arange(no_of_entries)
    np.random.shuffle(indx)
    shuffled_data = carPrice_data.iloc[indx]
    # split the shuffled data fram data into training(60%),validation(20%) and test (20%) datasets
    train_data = shuffled_data.iloc[:no_of_entries_train].copy()
    val_data = shuffled_data.iloc[no_of_entries_train:
                                  no_of_entries_train+no_of_entries_val].copy()
    test_data = shuffled_data.iloc[no_of_entries_train +
                                   no_of_entries_val:].copy()
    # Target Variables for each of the train,val,test data sets
    y_train_orginal = train_data.msrp.values
    y_val_original = val_data.msrp.values
    y_test_original = test_data.msrp.values
    # Applying log transformation .
    y_train = np.log1p(train_data.msrp.values)
    y_val = np.log1p(val_data.msrp.values)
    y_test = np.log1p(test_data.msrp.values)
    # Selecting five features in order to use in the Linear Regression model
    base = ['engine_hp', 'engine_cylinders',
            'highway_mpg', 'city_mpg', 'popularity']

    X_train = carprice_data.prepare_X(train_data, base)
    # Used the linear regression function (to get the weight params)
    w_0, w_1 = carprice_data.linear_regression(X_train, y_train)

    X_val = carprice_data.prepare_X(val_data, base)
    y_pred_val = w_0 + X_val.dot(w_1)
    print("For Validation data set, the RMSE value of Original msrp and predicted msrp is: ",
          carprice_data.validate(y_val, y_pred_val))
    X_test = carprice_data.prepare_X(test_data, base)
    y_pred_test = w_0 + X_val.dot(w_1)
    print("For Validation data set, the RMSE value of Original msrp and predicted msrp is: ",
          carprice_data.validate(y_test, y_pred_test))

    y_pred_MSRP_val = np.expm1(y_pred_val)

    val_data['msrp_pred'] = y_pred_MSRP_val

    print("Final Required Output")
    # displaying 5 cars with Original msrp and predicted msrp
    print(val_data.iloc[:, 5:].head().to_markdown(), "\n")


if __name__ == "__main__":
    # execute only if run as a script
    calculate()
