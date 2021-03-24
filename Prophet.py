import fbprophet
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta


class Prophet:
    """
    Works only for monthly data (month start index)
    """

    def __init__(self, y, horizon=None, ub=None, lb=None):
        """
        :param y: series with date index
        :param horizon: how many steps ahead model should predict values.
            If None, model will not use any information to create prediction besides train series.
        """

        self.train_series = y
        self.logistic_growth = lb is not None or ub is not None
        self.horizon = horizon
        self.ub = ub
        self.lb = lb

        self.model = None

    def fit(self):
        train_data = self._prepare_input_data(self.train_series)
        if self.logistic_growth:
            self.model = fbprophet.Prophet(growth='logistic', yearly_seasonality=True, weekly_seasonality=False,
                                           daily_seasonality=False)
        else:
            self.model = fbprophet.Prophet(yearly_seasonality=True, weekly_seasonality=False,
                                           daily_seasonality=False)
        self.model.fit(train_data)

    def predict(self, test_data, plot=True):
        self._check_fitted()

        future = pd.DataFrame(test_data.index, columns=['ds'])
        future = self._fill_data(future)
        forecast = self.model.predict(future).set_index('ds')

        if plot:
            ax = test_data.plot(label='True values')
            forecast.yhat.plot(ax=ax, label='Predicted values')

            ax.fill_between(forecast.index, forecast['yhat_lower'], forecast['yhat_upper'], color='k',
                            alpha=.2)
            plt.legend()
            plt.show()

        return forecast.yhat

    def analyse_results(self):
        pass

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
        if self.model is None:
            raise Exception("You should fit model first")
