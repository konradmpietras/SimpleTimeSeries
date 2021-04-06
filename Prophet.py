import fbprophet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics


class Prophet:
    """
    Works only for monthly data (month start index)
    """

    def __init__(self, y, horizon=None, ub=None, lb=None):
        """
        :param y: series with date index
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

        self.seasonality_mode = None
        self.changepoint_prior_scale = None

    def fit(self, seasonality_mode='multiplicative', changepoint_prior_scale=0.05):
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale

    def hiperparameter_search_fit(self, seasonality_modes=['additive', 'multiplicative'],
                                  changepoint_prior_scale_list=[0.001, 0.01, 0.05, 0.1, 0.5], verbose=1):

        best_metric_value = np.inf
        best_params = None

        train_data = self._prepare_input_data(self.train_series)
        for seasonality_mode in seasonality_modes:
            for changepoint_prior_scale in changepoint_prior_scale_list:
                model = self._get_clean_model(seasonality_mode=seasonality_mode,
                                              changepoint_prior_scale=changepoint_prior_scale)\
                    .fit(train_data)

                cutoffs = pd.date_range(start=self.train_series.index[12], end=self.train_series.index[-1], freq='2MS')
                df_cv = cross_validation(model, horizon=f'{self.horizon * 30} days', cutoffs=cutoffs)
                df_p = performance_metrics(df_cv, rolling_window=1)
                metric_value = df_p['rmse'].values[0]

                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_params = {'seasonality_mode': seasonality_mode,
                                   'changepoint_prior_scale': changepoint_prior_scale}

                    if verbose > 0:
                        print(f"  Found better parameters {best_params} with {metric} value of {best_metric_value}")

        self.fit(seasonality_mode=best_params['seasonality_mode'],
                 changepoint_prior_scale=best_params['changepoint_prior_scale'])

    def predict(self, test_data, plot=True):
        self._check_fitted()

        if self.horizon is not None:
            prediction = pd.Series(index=test_data.index, dtype=float)
            conf_intervals = pd.DataFrame(index=test_data.index, columns=['yhat_lower', 'yhat_upper'], dtype=float)

            for index in test_data.index:
                last_data_index = index + relativedelta(months=-self.horizon)

                horizon_input_data = self._concat_series(self.train_series, test_data.loc[:last_data_index])
                horizon_input_data = self._prepare_input_data(horizon_input_data)

                model = self._get_clean_model(seasonality_mode=self.seasonality_mode,
                                                 changepoint_prior_scale=self.changepoint_prior_scale)

                model.fit(horizon_input_data)

                future = pd.DataFrame([index], columns=['ds'])
                future = self._fill_data(future)
                forecast = self.model.predict(future).set_index('ds')

                conf_intervals.loc[index, ['yhat_lower', 'yhat_upper']] = \
                    forecast.loc[index, ['yhat_lower', 'yhat_upper']]
                prediction.loc[index] = forecast.loc[index, 'yhat']

        else:
            model = self._get_clean_model(seasonality_mode=self.seasonality_mode,
                                             changepoint_prior_scale=self.changepoint_prior_scale)
            train_data = self._prepare_input_data(self.train_series)
            model.fit(train_data)

            future = pd.DataFrame(test_data.index, columns=['ds'])
            future = self._fill_data(future)
            forecast = model.predict(future).set_index('ds')

            conf_intervals = forecast[['yhat_lower', 'yhat_upper']]
            prediction = forecast.yhat

        if plot:
            ax = test_data.plot(label='True values')
            prediction.plot(ax=ax, label='Predicted values')

            ax.fill_between(conf_intervals.index, conf_intervals.iloc[:, 0], conf_intervals.iloc[:, 1], color='k',
                            alpha=.2)
            plt.legend()
            plt.show()

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
        if self.logistic_growth:
            return fbprophet.Prophet(growth='logistic', weekly_seasonality=False, daily_seasonality=False,
                                     seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)
        else:
            return fbprophet.Prophet(weekly_seasonality=False, daily_seasonality=False,
                                     seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)

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