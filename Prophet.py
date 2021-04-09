import fbprophet
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error

from BaseForecaster import BaseForecaster


class Prophet(BaseForecaster):
    """
    Works only for monthly data (month start index)
    """

    def __init__(self, y, horizon=None, ub=None, lb=None):
        """
        :param y: series with monthly date index (MS)
        :param horizon: how many steps ahead model should predict values.
            If None, model will not use any information to create prediction besides train series.
        :param ub: upper bound for values in series
        :param lb:  lower bound for values in series
        """

        self.train_series = y
        self.logistic_growth = lb is not None or ub is not None
        self.horizon = horizon
        self.ub = ub
        self.lb = lb

        # model parameters
        self.seasonality_mode = None
        self.changepoint_prior_scale = None

    def fit(self, seasonality_mode='multiplicative', changepoint_prior_scale=0.05):
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale

    def hiperparameter_search_fit(self, seasonality_modes=['additive', 'multiplicative'],
                                  changepoint_prior_scale_list=[0.001, 0.01, 0.05, 0.1, 0.5],
                                  split_fraction=0.8, verbose=1):

        best_metric_value, best_params = np.inf, {}

        train_series, validation_series = self._train_val_split(split_fraction=split_fraction)

        for seasonality_mode in seasonality_modes:
            for changepoint_prior_scale in changepoint_prior_scale_list:
                model = Prophet(y=train_series, horizon=self.horizon, ub=self.ub, lb=self.lb)
                model.fit(seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)

                val_prediction = model.predict(test_data=validation_series, plot=False)
                metric_value = mean_squared_error(y_true=validation_series, y_pred=val_prediction)

                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_params = {'seasonality_mode': seasonality_mode,
                                   'changepoint_prior_scale': changepoint_prior_scale}
                    if verbose > 0:
                        print(f"  Found better parameters {best_params} with MSE value of {best_metric_value}")

        self.fit(seasonality_mode=best_params['seasonality_mode'],
                 changepoint_prior_scale=best_params['changepoint_prior_scale'])

    def predict(self, test_data, plot=True, verbose=0):
        self._check_fitted()

        prediction, conf_intervals = self._get_model_prediction(test_data=test_data, verbose=verbose)

        if plot:
            self.plot_predictions(true_values=test_data, prediction=prediction, conf_intervals=conf_intervals)

        return prediction

    def _prepare_input_data(self, input_data):
        input_data = input_data.reset_index()
        input_data.columns = ['ds', 'y']

        return self._fill_data(input_data)

    def _fill_data(self, data):
        if self.lb is not None:
            data['floor'] = self.lb
        if self.ub is not None:
            data['cap'] = self.ub

        return data

    def _check_fitted(self):
        if self.seasonality_mode is None or self.changepoint_prior_scale is None:
            raise Exception("You should fit model first")

    def _get_clean_model(self, seasonality_mode, changepoint_prior_scale):
        growth = 'logistic' if self.logistic_growth else 'linear'

        return fbprophet.Prophet(growth=growth, yearly_seasonality=True,
                                 weekly_seasonality=False, daily_seasonality=False,
                                 seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)

    @staticmethod
    def _concat_series(s1, s2):
        Prophet._check_overlapping(s1, s2)
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

    def _get_model_prediction(self, test_data, verbose):
        """
        returns prediction and confidence intervals tuple
        """

        if self.horizon is not None:
            return self._predict_with_horizon(test_data=test_data, horizon=self.horizon, verbose=verbose)
        else:
            forecast = self._predict_with_retrained_model(train_data=self.train_series,
                                                          prediction_index=test_data.index, verbose=verbose)

            return forecast.yhat, forecast[['yhat_lower', 'yhat_upper']]

    def _predict_with_horizon(self, test_data, horizon, verbose):
        """
        iteratively filling training data so it uses maximum range of data for given horizon making prediction for some
        month
        """

        prediction = pd.Series(index=test_data.index, dtype=float)
        conf_intervals = pd.DataFrame(index=test_data.index, columns=['yhat_lower', 'yhat_upper'], dtype=float)

        for index in test_data.index:
            last_input_index = index + relativedelta(months=-horizon)
            horizon_input_data = self._concat_series(self.train_series, test_data.loc[:last_input_index])

            forecast = self._predict_with_retrained_model(train_data=horizon_input_data, prediction_index=[index],
                                                          verbose=verbose)

            conf_intervals.loc[index, ['yhat_lower', 'yhat_upper']] = \
                forecast.loc[index, ['yhat_lower', 'yhat_upper']]
            prediction.loc[index] = forecast.loc[index, 'yhat']

        return prediction, conf_intervals

    def _predict_with_retrained_model(self, train_data, prediction_index, verbose):
        model = self._get_clean_model(seasonality_mode=self.seasonality_mode,
                                      changepoint_prior_scale=self.changepoint_prior_scale)

        train_data = self._prepare_input_data(train_data)
        if verbose == 0:
            with suppress_stdout_stderr():
                model.fit(train_data)
        else:
            model.fit(train_data)

        future = pd.DataFrame(prediction_index, columns=['ds'])
        future = self._fill_data(future)
        return model.predict(future).set_index('ds')


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
