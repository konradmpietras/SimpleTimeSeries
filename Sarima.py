from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta

from BaseModel import BaseModel


class Sarima(BaseModel):
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

        self.order_params = None
        self.seasonal_order_params = None

    def plot_acf(self):
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(self.train_series)
        print("If on ACF plot there is a positive autocorrelation at lag 1 and plot is decaing towards 0 ->"
              " use AR model (q=0)\n"
              "If on ACF plot there is a negative autocorrelation at lag 1 and plot drops sharply after few lags ->"
              " use MA model (p=0)")

    def plot_pacf(self):
        from statsmodels.graphics.tsaplots import plot_pacf

        plot_pacf(self.train_series)
        print("If PACF plot drops off at lag n -> use p=n, q=0\n"
              "If PACF plot drop is more gradual -> use MA term (p=0)")

    def hiperparameter_search_fit(self, metric, split_fraction=0.8,  p=[0, 1, 2], d=[0, 1, 2], q=[0, 1, 2], P=[0, 1],
                              D=[0, 1], Q=[0, 1], s=12, verbose=1):
        """
        :param metric: one of 'AIC', 'BIC', 'mse'. Metric used to choose best set of parameters
        :param split_fraction: used in case of mteric 'mse' to create validation set to assess accuracy. 0.8 means that
            first 80% of data is used as training data and rest as validation set
        :param p: list of all values to check
        :param d: list of all values to check
        :param q: list of all values to check
        :param P: list of all values to check
        :param D: list of all values to check
        :param Q: list of all values to check
        :param s: cycle lenght. For monthly data it means how many months is one season.
        :param verbose: 0 means no additional information is printed on screen
        """

        search_space = self._get_search_space(p_list=p, d_list=d, q_list=q, P_list=P, D_list=D, Q_list=Q, s=s)

        if metric in ['AIC', 'BIC']:
            self.order_params, self.seasonal_order_params = \
                self._information_criterion_search(metric=metric, search_space=search_space, verbose=verbose)
        elif metric == 'mse':
            self.order_params, self.seasonal_order_params = \
                self._validation_based_search(split_fraction=split_fraction, metric=metric, search_space=search_space,
                                              verbose=verbose)
        else:
            raise Exception(f"Invalid metric param value {metric}\n Allowed are 'AIC', 'BIC', 'mse'")

    def fit(self, order, seasonal_order):
        if len(order) != 3:
            raise Exception("Parameter order should be a tuple (p, d, q) where:\n "
                            "p is a number of previous steps to consider\n "
                            "d is a number of differencing steps to ensure stationarity\n "
                            "q is a number of previous prediction errors to consider")

        if len(seasonal_order) != 4:
            raise Exception("Parameter order should be a tuple (P, D, Q, s) where:\n "
                            "P is a number of previous steps to consider for seasonal component\n "
                            "D is a number of differencing steps to ensure stationarity for seasonal component\n "
                            "Q is a number of previous prediction errors to consider for seasonal component\n "
                            "s is a number od periods in season")

        self.order_params = order
        self.seasonal_order_params = seasonal_order

    def predict(self, test_data, plot=True):
        self._check_fitted()

        if self.horizon is not None:
            prediction = pd.Series(index=test_data.index, dtype=float)
            conf_intervals = pd.DataFrame(index=test_data.index, columns=['lower y', 'upper y'], dtype=float)

            for index in test_data.index:
                last_data_index = index + relativedelta(months=-self.horizon)

                horizon_input_data = self._concat_series(self.train_series, test_data.loc[:last_data_index])
                model = Sarima._get_clean_model(y=horizon_input_data, order=self.order_params,
                                                seasonal_order=self.seasonal_order_params)
                results = model.fit(disp=-1)
                prediction_data = results.get_prediction(start=index, end=index)

                conf_intervals.loc[index, ['lower y', 'upper y']] = prediction_data.conf_int().loc[index]
                prediction.loc[index] = prediction_data.predicted_mean.loc[index]

        else:
            model = Sarima._get_clean_model(y=self.train_series, order=self.order_params, seasonal_order=self.seasonal_order_params)
            results = model.fit(disp=-1)

            prediction_data = results.get_prediction(start=test_data.index[0], end=test_data.index[-1])
            conf_intervals = prediction_data.conf_int()
            prediction = prediction_data.predicted_mean

        if plot:
            self.plot_predictions(true_values=test_data, prediction=prediction, conf_intervals=conf_intervals)

        return prediction

    def analyse_results(self):
        self._check_fitted()

        model = Sarima._get_clean_model(y=self.train_series, order=self.order_params,
                                        seasonal_order=self.seasonal_order_params)

        results = model.fit(disp=-1)
        results.plot_diagnostics(figsize=(15, 12))
        plt.show()

        print(results.summary())

    def _check_fitted(self):
        if self.order_params is None or self.seasonal_order_params is None:
            raise Exception("You should fit model first")

    @staticmethod
    def _get_search_space(p_list, d_list, q_list, P_list, D_list, Q_list, s):
        order_space = itertools.product(p_list, d_list, q_list)
        seasonal_order_space = itertools.product(P_list, D_list, Q_list, [s])
        return list(itertools.product(order_space, seasonal_order_space))

    def _information_criterion_search(self, metric, search_space, verbose):
        best_metric_value = np.inf
        best_params = None

        for order, seasonal_order in search_space:
            model = self._get_clean_model(y=self.train_series, order=order, seasonal_order=seasonal_order)
            results = model.fit(disp=-1)

            if metric == 'AIC':
                metric_value = results.aic
            elif metric == 'BIC':
                metric_value = results.bic
            else:
                raise Exception(f"Invalid information criterion {metric}\n Allowed are 'AIC', 'BIC'")

            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_params = order, seasonal_order

                if verbose > 0:
                    print(f"  Found better parameters {best_params} with {metric} value of {best_metric_value}")

        return best_params

    def _train_val_split(self, split_fraction):
        split_point = split_fraction * len(self.train_series)
        return self.train_series.iloc[:split_point], self.train_series.iloc[split_point:]

    def _validation_based_search(self, split_fraction, metric, search_space, verbose):
        best_metric_value = np.inf
        best_params = None

        train_series, validation_series = self._train_val_split(split_fraction=split_fraction)

        model = Sarima(y=train_series, horizon=self.horizon)

        for order, seasonal_order in search_space:
            model.fit(order=order, seasonal_order=seasonal_order)

            val_prediction = model.predict(test_data=validation_series, plot=False)

            if metric == 'mse':
                metric_value = mean_squared_error(y_true=validation_series, y_pred=val_prediction)
            else:
                raise Exception(f"Invalid criterion {metric}\n Allowed is 'mse'")

            if metric_value < best_metric_value:
                best_metric_value = metric_value
                best_params = order, seasonal_order

                if verbose > 0:
                    print(f"  Found better parameters {best_params} with {metric} value of {best_metric_value}")

        return best_params

    @staticmethod
    def _get_clean_model(y, order, seasonal_order):
        return SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=True,
                       enforce_invertibility=True, disp=False)

    @staticmethod
    def _concat_series(s1, s2):
        intersected_index = list(set(s1.index).intersection(s2.index))

        if len(intersected_index) > 0:
            if not s1.loc[intersected_index].equals(s2.loc[intersected_index]):
                raise Exception(f"Different values for the same timesteps {s1.loc[intersected_index]} and "
                                f"{s2.loc[intersected_index]}")

        s2 = s2.loc[s1.index[-1] + relativedelta(months=1):]

        if len(s2) == 0:
            return s1

        full_index = pd.date_range(start=s1.index[0], end=s2.index[-1], freq='MS')

        return pd.concat([s1, s2]).reindex(index=full_index)
