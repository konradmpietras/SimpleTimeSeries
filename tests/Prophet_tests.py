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
    model.fit(seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
    model.predict(test_data=train_series)


def predict_with_train_series_h4():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series, horizon=4)
    model.fit(seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
    model.predict(test_data=train_series)


def predict_with_intersected_series_h1():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series.iloc[:15], horizon=1)
    model.fit(seasonality_mode='additive', changepoint_prior_scale=0.001)
    model.predict(test_data=train_series.iloc[10:])


def predict_with_intersected_series_hNone():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series.iloc[:15], horizon=None)
    model.fit(seasonality_mode='multiplicative', changepoint_prior_scale=0.01)
    model.predict(test_data=train_series.iloc[10:])


def predict_with_intersected_series_with_different_values():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series.iloc[:15], horizon=1)
    model.fit(seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
    try:
        model.predict(test_data=train_series.iloc[10:].shift(-1))
    except Exception:
        print("Exception, OK")
        return True

    raise Exception("Validation should have failed")


def predict_with_new_series_h3():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series.iloc[:12], horizon=3)
    model.fit(seasonality_mode='multiplicative', changepoint_prior_scale=0.005)
    model.predict(test_data=train_series.iloc[13:])


def predict_with_new_series_hNone():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series.iloc[:12], horizon=None)
    model.fit(seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
    model.predict(test_data=train_series.iloc[13:])


def compare_incremantal_prediction():
    train_series = pd.Series(get_ar2_process_values(n_samples=24),
                             index=pd.date_range('2013-01-01', freq='MS', periods=24))

    model = Prophet(y=train_series.iloc[:12], horizon=4)
    model.fit(seasonality_mode='additive', changepoint_prior_scale=0.001)
    test_series = train_series.iloc[12:]
    normal_prediction = model.predict(test_data=test_series, plot=False)

    for index in test_series.index[4:]:
        model = Prophet(y=train_series.iloc[:12], horizon=4)
        model.fit(seasonality_mode='additive', changepoint_prior_scale=0.001)

        last_index = index - relativedelta(months=4)
        iter_test_data = train_series.loc[:last_index]
        result_index = pd.date_range(iter_test_data.index.min(), index, freq='MS')
        iter_test_data = iter_test_data.reindex(result_index)
        inc_pred = model.predict(iter_test_data, plot=False).loc[index]

        print(f"Should be {normal_prediction.loc[index]}. Got {inc_pred} "
              f"({np.isclose(normal_prediction.loc[index], inc_pred)})\n")


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

# predict_with_train_series_h1()
# predict_with_train_series_h4()

# predict_with_intersected_series_h1()
# predict_with_intersected_series_hNone()
# predict_with_intersected_series_with_different_values()

# predict_with_new_series_h3()
# predict_with_new_series_hNone()
compare_incremantal_prediction()
