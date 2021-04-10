import matplotlib.pyplot as plt
import statsmodels.api as sm


def seasonal_decompose(y):
    import statsmodels.api as sm

    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 12

    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    decomposition.plot()
    plt.show()


def plot_differenced_series(y, d):
    for _ in range(d):
        y = y.diff()

    y.plot()
    plt.show()


def test_if_stationary(y, conf_level=0.05):
    p_value = sm.tsa.stattools.adfuller(y)[1]

    print(f"Received p-value of {p_value}")
    if p_value > conf_level:
        print("Null hypothesis can not be rejected. Series seems to be non-stationary")
    else:
        print("Null hypothesis is rejected. Series seems to be stationary")
