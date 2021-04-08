import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


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

    def fit(self, model_type, trend, seasonal):
        self.model_type = model_type
        self.trend = trend
        self.seasonal = seasonal

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

    def decompose_train_data(self, model_type):
        if model_type == 'add':
            model = 'additive'
        elif model_type == 'mul':
            model = 'multiplicative'
        else:
            raise Exception(f"Invalid value for model_type parameter ({model_type})\nAllowed are: 'add' and 'mul'")
        seasonal_decompose(self.train_series, model=model)

    def _get_model_prediction(self, test_data):
        pass

    def _check_fitted(self):
        pass

