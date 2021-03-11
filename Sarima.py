from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error

# TODO przeczytać https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_internet.html
# TODO dorobienie rozróżnienia predykcji one-step-ahead i dłuższej


class Sarima:
    def __init__(self, y):
        self.y = y
        self.results = None

    def plot_acf(self):
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(self.y)
        print("If on ACF plot there is a positive autocorrelation at lag 1 and plot is decaing towards 0 ->"
              " use AR model (q=0)\n"
              "If on ACF plot there is a negative autocorrelation at lag 1 and plot drops sharply after few lags ->"
              " use MA model (p=0)")

    def plot_pacf(self):
        from statsmodels.graphics.tsaplots import plot_pacf

        plot_pacf(self.y)
        print("If PACF plot drops off at lag n -> use p=n, q=0\n"
              "If PACF plot drop is more gradual -> use MA term (p=0)")

    def hiperparameter_search(self, metric, split_fraction=0.8,  p=[0, 1, 2], d=[0, 1, 2], q=[0, 1, 2], P=[0, 1],
                              D=[0, 1], Q=[0, 1], s=12, verbose=1):
        order_space = itertools.product(p, d, q)
        seasonal_order_space = itertools.product(P, D, Q, [s])
        search_space = list(itertools.product(order_space, seasonal_order_space))

        if metric in ['AIC', 'BIC']:
            return self._information_criterion_search(metric=metric, search_space=search_space, verbose=verbose)
        elif metric == 'mse':
            return self._validation_based_search(split_fraction=split_fraction, metric=metric,
                                                 search_space=search_space, verbose=verbose)
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

        model = self._get_clean_model(y=self.y, order=order, seasonal_order=seasonal_order)

        self.results = model.fit()

    def predict(self, start_index, end_index, plot=True):
        self._check_fitted()

        prediction = self.results.get_prediction(start=start_index, end=end_index)

        if plot:
            conf_intervals = prediction.conf_int()

            ax = self.y.plot(label='True values')
            prediction.predicted_mean.plot(ax=ax, label='Predicted values')
            ax.fill_between(conf_intervals.index, conf_intervals.iloc[:, 0], conf_intervals.iloc[:, 1], color='k',
                            alpha=.2)
            plt.legend()

            plt.show()

        return prediction.predicted_mean

    def analyse_results(self):
        self._check_fitted()

        self.results.plot_diagnostics(figsize=(15, 12))
        plt.show()

        print(self.results.summary())

    def _check_fitted(self):
        if self.results is None:
            raise Exception("You should fit model first")

    def _information_criterion_search(self, metric, search_space, verbose):
        best_matric_value = np.inf
        best_params = None

        for order, seasonal_order in search_space:
            model = self._get_clean_model(y=self.y, order=order, seasonal_order=seasonal_order)
            results = model.fit()

            if metric == 'AIC':
                metric_value = results.aic
            elif metric == 'BIC':
                metric_value = results.bic
            else:
                raise Exception(f"Invalid information criterion {metric}\n Allowed are 'AIC', 'BIC'")

            if metric_value < best_matric_value:
                best_matric_value = metric_value
                best_params = order, seasonal_order

                if verbose > 0:
                    print(f"  Found better parameters {best_params} with {metric} value of {best_matric_value}")

        return best_params

    def _train_val_split(self, split_fraction):
        split_point = split_fraction * len(self.y)
        return self.y.iloc[:split_point], self.y.iloc[split_point:]

    def _validation_based_search(self, split_fraction, metric, search_space, verbose):
        best_matric_value = np.inf
        best_params = None

        train_series, validation_series = self._train_val_split(split_fraction=split_fraction)

        for order, seasonal_order in search_space:
            model = self._get_clean_model(y=train_series, order=order, seasonal_order=seasonal_order)
            results = model.fit()

            val_prediction = results.get_prediction(start=validation_series.index[0], end=validation_series.index[-1])

            if metric == 'mse':
                metric_value = mean_squared_error(y_true=validation_series, y_pred=val_prediction)
            else:
                raise Exception(f"Invalid criterion {metric}\n Allowed is 'mse'")

            if metric_value < best_matric_value:
                best_matric_value = metric_value
                best_params = order, seasonal_order

                if verbose > 0:
                    print(f"  Found better parameters {best_params} with {metric} value of {best_matric_value}")

        return best_params

    def _get_clean_model(self, y, order, seasonal_order):
        return SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=True,
                        enforce_invertibility=True)