import matplotlib.pyplot as plt


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