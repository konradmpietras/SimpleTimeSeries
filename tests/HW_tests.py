import pandas as pd
import numpy as np
from HW import HoltWinters
from dateutil.relativedelta import relativedelta


def predict_with_train_series():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = HoltWinters(y=train_series)
    model.fit(model_type='triple', trend='add', seasonal='add')
    model.predict(test_data=train_series)


def predict_with_intersected_series():
    train_series = pd.Series(get_ar2_process_values(n_samples=36),
                             index=pd.date_range('2013-01-01', freq='MS', periods=36))

    model = HoltWinters(y=train_series.iloc[:24], horizon=None)
    model.fit(model_type='triple', trend='add', seasonal='add')
    model.predict(test_data=train_series.iloc[20:])


def predict_with_intersected_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=36),
                             index=pd.date_range('2013-01-01', freq='MS', periods=36))

    model = HoltWinters(y=train_series.iloc[:24], horizon=1)
    model.fit(model_type='triple', trend='add', seasonal='add')
    model.predict(test_data=train_series.iloc[20:])


def hiperfit_with_intersected_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=50),
                             index=pd.date_range('2013-01-01', freq='MS', periods=50))

    model = HoltWinters(y=train_series.iloc[:40], horizon=1)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series.iloc[35:])


def get_ar2_process_values(n_samples):
    ret = []

    a = 5
    ret.append(a)
    b = 3
    ret.append(b)

    for _ in range(n_samples - 2):
        next = 0.8 * a + 0.2 * b
        a, b = b, next

        ret.append(next)

    return ret


# predict_with_train_series()
# predict_with_intersected_series()
# predict_with_intersected_series_h1()
hiperfit_with_intersected_series_h1()