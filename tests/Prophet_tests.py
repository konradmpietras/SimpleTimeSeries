import pandas as pd
import numpy as np

from Prophet import Prophet
from dateutil.relativedelta import relativedelta


def predict_with_train_series():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series)
    model.fit()
    model.predict(test_data=train_series)


def predict_with_train_series_with_bounds():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series, lb=0, ub=3)
    model.fit()
    model.predict(test_data=train_series)


def predict_with_train_series_with_single_bound():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series, ub=3)
    model.fit()
    model.predict(test_data=train_series)


def predict_with_train_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series, horizon=1)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series)


def get_ar2_process_values(n_samples):
    ret = []

    a = 5
    ret.append(a)
    b = 3
    ret.append(b)

    for _ in range(n_samples - 2):
        next = 0.8 * a - 0.2 * b
        a, b = b, next

        ret.append(next)

    return ret


# predict_with_train_series()
# predict_with_train_series_with_bounds()
# predict_with_train_series_with_single_bound()
predict_with_train_series_h1()