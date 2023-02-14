import xgboost
import numpy as np


def run_xgboost_100(X, y, X_validation, y_validation):
    model = xgboost.XGBRegressor(n_estimators=100)
    model.fit(X, y)
    rmse = np.sqrt(((model.predict(X_validation) - y_validation) ** 2).mean())
    return rmse


import sklearn.linear_model


def run_lasso(X, y, X_validation, y_validation):
    X = np.log(X)
    X_validation = np.log(X_validation)
    shift, scale = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    X_validation = (X_validation - shift) / scale
    X = (X - shift) / scale

    model = sklearn.linear_model.Lasso(alpha=0.05)
    model.fit(X, y)
    rmse = np.sqrt(((model.predict(X_validation) - y_validation) ** 2).mean())
    return rmse


def run_lm(X, y, X_validation, y_validation):
    X = np.log(X)
    X_validation = np.log(X_validation)
    shift, scale = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    X_validation = (X_validation - shift) / scale
    X = (X - shift) / scale

    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    rmse = np.sqrt(((model.predict(X_validation) - y_validation) ** 2).mean())
    return rmse


import sklearn.svm


def run_svr(X, y, X_validation, y_validation):
    # X = np.log(X)
    # X_validation = np.log(X_validation)
    # shift, scale = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    # X_validation = (X_validation - shift) / scale
    # X = (X - shift) / scale1

    model = sklearn.svm.SVR()
    model.fit(X, y)
    rmse = np.sqrt(((model.predict(X_validation) - y_validation) ** 2).mean())
    return rmse


predictor_funcs = {
    "xgboost_100": run_xgboost_100,
    "lasso": run_lasso,
    "lm": run_lm,
    "svr": run_svr,
}
