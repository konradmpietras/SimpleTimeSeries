import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWinters:
    """
    Works only for monthly data (month start index)
    """

    def __init__(self, y, horizon=None):
        """
        :param y: series with date index
        :param horizon: how many steps ahead model should predict values.
            If None, model will not use any information to create prediction besides train series.
        """

        self.train_series = y
        self.horizon = horizon

        self.model_type = None
        self.trend = None
        self.seasonal = None

    def decompose_train_data(self, model_type):
        if model_type == 'add':
            model = 'additive'
        elif model_type == 'mul':
            model = 'multiplicative'
        else:
            raise Exception(f"Invalid value for model_type parameter ({model_type})\nAllowed are: 'add' and 'mul'")
        seasonal_decompose(self.train_series, model=model)

    def fit(self, model_type, trend, seasonal):
        self.model_type = model_type
        self.trend = trend
        self.seasonal = seasonal

    def hiperparameter_search_fit(self, split_fraction=0.8, model_types=['single', 'double', 'triple'],
                                  trends=['add', 'mul'], seasonals=['add', 'mul'], verbose=1):

        best_metric_value, best_params = np.inf, {}

        train_series, validation_series = self._train_val_split(split_fraction=split_fraction)

        for model_type in model_types:
            for trend in trends:
                for seasonal in seasonals:
                    model = HoltWinters(y=train_series, horizon=self.horizon)
                    model.fit(model_type=model_type, trend=trend, seasonal=seasonal)

                    val_prediction = model.predict(test_data=validation_series, plot=False)
                    metric_value = mean_squared_error(y_true=validation_series, y_pred=val_prediction)

                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_params = {'model_type': model_type,
                                       'trend': trend,
                                       'seasonal': seasonal
                                       }
                        if verbose > 0:
                            print(f"  Found better parameters {best_params} with MSE value of {best_metric_value}")

        self.fit(model_type=best_params['model_type'],
                 trend=best_params['trend'],
                 seasonal=best_params['seasonal'])

    def predict(self, test_data, plot=True):
        self._check_fitted()

        prediction, conf_intervals = self._get_model_prediction(test_data=test_data)

        if plot:
            ax = test_data.plot(label='True values')
            prediction.plot(ax=ax, label='Predicted values')

            ax.fill_between(conf_intervals.index, conf_intervals.iloc[:, 0], conf_intervals.iloc[:, 1], color='k',
                            alpha=.2)
            plt.legend()
            plt.show()

        return prediction

    def _get_model_prediction(self, test_data):
        """
        returns prediction and confidence intervals tuple
        """

        if self.horizon is not None:
            return self._predict_with_horizon(test_data=test_data, horizon=self.horizon)
        else:
            return self._predict_with_retrained_model(train_data=self.train_series, prediction_index=test_data.index)

    def _check_fitted(self):
        if self.model_type is None or self.trend is None or self.seasonal is None:
            raise Exception("You should fit model first")

    def _predict_with_horizon(self, test_data, horizon):
        """
         iteratively filling training data so it uses maximum range of data for given horizon making prediction for some
         month
         """

        prediction = pd.Series(index=test_data.index, dtype=float)
        conf_intervals = pd.DataFrame(index=test_data.index, columns=['lower', 'upper'], dtype=float)

        for index in test_data.index:
            last_input_index = index + relativedelta(months=-horizon)
            horizon_input_data = self._concat_series(self.train_series, test_data.loc[:last_input_index])

            prediction, conf_intervals = self._predict_with_retrained_model(train_data=horizon_input_data,
                                                                            prediction_index=[index])

            conf_intervals.loc[index, ['lower', 'upper']] = conf_intervals.loc[index, ['lower', 'upper']]
            prediction.loc[index] = prediction.loc[index]

        return prediction, conf_intervals

    def _predict_with_retrained_model(self, train_data, prediction_index):
        model = self._get_clean_model(train_data=train_data, model_type=self.model_type, trend=self.trend,
                                      seasonal=self.seasonal)

        fitted_model = model.fit()

        train_prediction = pd.Series(fitted_model.fittedvalues, index=train_data.index)

        after_train_index = pd.date_range(train_data.index[-1] + relativedelta(months=1), prediction_index[-1])
        after_train_prediction = pd.Series(fitted_model.forecast(len(after_train_index)), index=after_train_index)

        model_prediction = pd.concat([train_prediction, after_train_prediction])

        return model_prediction.loc[prediction_index]

    @staticmethod
    def _concat_series(s1, s2):
        HoltWinters._check_overlapping(s1, s2)
        s2 = s2.loc[s1.index[-1] + relativedelta(months=1):]

        if len(s2) == 0:
            return s1

        full_index = pd.date_range(start=s1.index[0], end=s2.index[-1], freq='MS')

        return pd.concat([s1, s2]).reindex(index=full_index)

    @staticmethod
    def _check_overlapping(s1, s2):
        intersected_index = list(set(s1.index).intersection(s2.index))

        if len(intersected_index) > 0:
            if not s1.loc[intersected_index].equals(s2.loc[intersected_index]):
                raise Exception(f"Different values for the same timesteps {s1.loc[intersected_index]} and "
                                f"{s2.loc[intersected_index]}")

    def _train_val_split(self, split_fraction):
        split_point = int(split_fraction * len(self.train_series))
        return self.train_series.iloc[:split_point], self.train_series.iloc[split_point:]

    def _get_clean_model(self, train_data, model_type, trend, seasonal):
        if model_type == 'single':
            return SimpleExpSmoothing(train_data)
        elif model_type == 'double':
            return ExponentialSmoothing(train_data, trend=trend)
        elif model_type == 'triple':
            return ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=12)
        else:
            raise Exception(f"Invalid value of model_type parameter ({model_type}).\n"
                            "Allowed are: 'single', 'double' or 'triple'")

