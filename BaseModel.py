import matplotlib.pyplot as plt


class BaseModel:

    @staticmethod
    def plot_predictions(true_values, prediction, conf_intervals=None):
        plt.figure(figsize=(16, 10))

        ax1 = plt.subplot(2, 1, 1)
        true_values.plot(label='True values', color='blue')
        prediction.plot(label='Predicted values', color='C1', style='--')
        if conf_intervals is not None:
            ax1.fill_between(conf_intervals.index, conf_intervals.iloc[:, 0], conf_intervals.iloc[:, 1], color='k',
                             alpha=.2)
        plt.legend()

        ax2 = plt.subplot(2, 2, 3)
        true_values.plot(label='True values', color='blue')
        plt.legend()

        plt.subplot(2, 2, 4, sharey=ax2)
        prediction.plot(label='Predicted values', color='C1')
        plt.legend()

        plt.show()
