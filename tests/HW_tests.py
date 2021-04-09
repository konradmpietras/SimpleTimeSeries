import pandas as pd
import numpy as np
from HW import HoltWinters
from dateutil.relativedelta import relativedelta


def predict_with_train_series():
    train_series = pd.Series(get_ar2_process_values(n_samples=48),
                             index=pd.date_range('2013-01-01', freq='MS', periods=48))

    model = HoltWinters(y=train_series)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series)


def predict_with_train_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=48),
                             index=pd.date_range('2013-01-01', freq='MS', periods=48))

    model = HoltWinters(y=train_series, horizon=1)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series)


def predict_with_train_series_h4():
    train_series = pd.Series(get_ar2_process_values(n_samples=48),
                             index=pd.date_range('2013-01-01', freq='MS', periods=48))

    model = HoltWinters(y=train_series, horizon=4)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series)


def predict_with_intersected_series():
    train_series = pd.Series(get_ar2_process_values(n_samples=36),
                             index=pd.date_range('2013-01-01', freq='MS', periods=36))

    model = HoltWinters(y=train_series.iloc[:24], horizon=None)
    model.fit(model_type='triple', trend_type='add', seasonal_type='add')
    model.predict(test_data=train_series.iloc[20:])


def predict_with_intersected_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=36),
                             index=pd.date_range('2013-01-01', freq='MS', periods=36))

    model = HoltWinters(y=train_series.iloc[:40], horizon=1)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series.iloc[20:])


def predict_with_intersected_series_h4():
    train_series = pd.Series(get_ar2_process_values(n_samples=40),
                             index=pd.date_range('2013-01-01', freq='MS', periods=40))

    model = HoltWinters(y=train_series.iloc[:30], horizon=4)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series.iloc[20:])


def predict_with_intersected_series_hNone():
    train_series = pd.Series(get_ar2_process_values(n_samples=50),
                             index=pd.date_range('2013-01-01', freq='MS', periods=50))

    model = HoltWinters(y=train_series.iloc[:30], horizon=None)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series)


def hiperfit_with_intersected_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=50),
                             index=pd.date_range('2013-01-01', freq='MS', periods=50))

    model = HoltWinters(y=train_series.iloc[:40], horizon=1)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series.iloc[35:])


def predict_with_new_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=100),
                             index=pd.date_range('2013-01-01', freq='MS', periods=100))

    model = HoltWinters(y=train_series.iloc[:30], horizon=1)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series.iloc[30:])


def predict_with_new_series_h3():
    train_series = pd.Series(get_ar2_process_values(n_samples=100),
                             index=pd.date_range('2013-01-01', freq='MS', periods=100))

    model = HoltWinters(y=train_series.iloc[:30], horizon=3)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series.iloc[30:])


def predict_with_new_series_hNone():
    train_series = pd.Series(get_ar2_process_values(n_samples=100),
                             index=pd.date_range('2013-01-01', freq='MS', periods=100))

    model = HoltWinters(y=train_series.iloc[:30], horizon=None)
    model.hiperparameter_search_fit()
    model.predict(test_data=train_series.iloc[30:])


def compare_incremantal_prediction():
    train_series = pd.Series(get_ar2_process_values(n_samples=40),
                             index=pd.date_range('2013-01-01', freq='MS', periods=40))

    model = HoltWinters(y=train_series.iloc[:30], horizon=4)
    model.fit(model_type='triple', trend_type='additive', seasonal_type='additive')
    test_series = train_series.iloc[30:]
    normal_prediction = model.predict(test_data=test_series, plot=False)

    for index in test_series.index[4:]:
        last_index = index - relativedelta(months=4)
        iter_test_data = train_series.loc[:last_index]
        result_index = pd.date_range(iter_test_data.index.min(), index, freq='MS')
        iter_test_data = iter_test_data.reindex(result_index)
        inc_pred = model.predict(iter_test_data, plot=False).loc[index]

        print(f"Should be {normal_prediction.loc[index]}. Got {inc_pred} " \
               f"({np.isclose(normal_prediction.loc[index], inc_pred)})\n")


def get_ar2_process_values(n_samples):
    ret = []

    a = 10
    ret.append(a)
    b = 9
    ret.append(b)

    for _ in range(n_samples - 2):
        next = 0.9 * a + 0.3 * b
        a, b = b, next

        ret.append(next)

    return ret


# predict_with_train_series()
# predict_with_train_series_h1()
# predict_with_train_series_h4()

# predict_with_intersected_series()
# hiperfit_with_intersected_series_h1()
# predict_with_intersected_series_h1()
# predict_with_intersected_series_h4()
# predict_with_intersected_series_hNone()

# predict_with_new_series_h1()
# predict_with_new_series_h3()
# predict_with_new_series_hNone()
#
compare_incremantal_prediction()
