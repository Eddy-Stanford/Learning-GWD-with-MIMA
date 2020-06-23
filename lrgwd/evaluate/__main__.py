import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from tensorflow import keras
from tensorflow.keras import layers

xmin = 400
xmax = 500
# Load Test Data
test = pd.read_csv("./small_data_dir/test_data.csv", sep=",")[xmin:xmax]
test_labels = pd.read_csv("./small_data_dir/test_labels.csv", sep=",")[xmin:xmax]
# Load Train Data
train = pd.read_csv("./small_data_dir/train_data.csv", sep=",")[xmin:xmax]
train_labels = pd.read_csv("./small_data_dir/train_labels.csv", sep=",")[xmin:xmax]
# train = pd.read_csv('../sdata/train_data.csv', sep=',')
# train_labels = pd.read_csv('./data/train_labels.csv', sep=',')

# Load Model
model = keras.models.load_model("./time_linear_model_v2")

# These values are scaled up due to initial data processing:
# either scale down or ignore
# loss, mae, mse = model.evaluate(test, test_labels, verbose=2)


def scale_labels(predictions, labels):
    # Correct values by scaling down
    predictions = predictions * (10 ** -7)
    labels = labels * (10 ** -7)
    predictions = np.reshape(np.array(predictions), predictions.shape[0], 1)
    labels = np.array(labels)

    return predictions, labels


# Test Model
test_predictions = model.predict(test)
train_predictions = model.predict(train)

test_predictions, test_labels = scale_labels(test_predictions, test_labels)
train_predictions, train_labels = scale_labels(train_predictions, train_labels)

# Evaluate Model
# mae = np.mean(np.absolute(np.subtract(test_labels, test_predictions)))
# print("-----------------------------")
# print("Mean Absolute Error: ", mae)
# print("-----------------------------")


def grid_plot():
    x = range(0, 5)
    y = range(5, 10)
    nlabels = test_labels.shape[0]

    truth_color = "#ef8a62"  # "#a8ddb5"
    prediction_color = "#43a2ca"  # "#67a9cf"
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(test_labels, range(nlabels), truth_color, label="truth")
    axs[0, 0].plot(
        test_predictions, range(nlabels), prediction_color, label="predictions"
    )
    axs[0, 0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axs[0, 0].set_title("Train Results")
    axs[0, 0].legend(loc="best")
    axs[0, 1].plot(train_labels, range(nlabels), truth_color, label="truth")
    axs[0, 1].plot(
        train_predictions, range(nlabels), prediction_color, label="predictions"
    )
    axs[0, 1].legend(loc="best")
    axs[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axs[0, 1].set_title("Test Results")

    # AXS [1, 0]
    axs[1, 0].scatter(test_predictions, test_labels, c=prediction_color)
    max_scatter = 0.000125
    min_scatter = -0.0000125
    lims = [min_scatter, max_scatter]
    axs[1, 0].set_xlim(lims)
    axs[1, 0].set_ylim(lims)

    num_major_locators = 5
    major_intervals = max_scatter / num_major_locators
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(major_intervals))
    axs[1, 0].yaxis.set_major_locator(MultipleLocator(major_intervals))
    axs[1, 0].xaxis.set_minor_locator(AutoMinorLocator(5))
    axs[1, 0].yaxis.set_minor_locator(AutoMinorLocator(5))
    axs[1, 0].grid(which="major", color="#CCCCCC", linestyle="--")
    axs[1, 0].grid(which="minor", color="#CCCCCC", linestyle=":")
    axs[1, 0].ticklabel_format(style="sci", scilimits=(0, 0))

    axs[1, 0].plot(lims, lims, c=truth_color)  # Straight Line

    # AXS[1,1]
    axs[1, 1].scatter(train_predictions, train_labels, c=prediction_color)
    max_scatter = 0.000125
    min_scatter = -0.0000125
    lims = [min_scatter, max_scatter]
    axs[1, 1].set_xlim(lims)
    axs[1, 1].set_ylim(lims)

    num_major_locators = 5
    major_intervals = max_scatter / num_major_locators
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(major_intervals))
    axs[1, 1].yaxis.set_major_locator(MultipleLocator(major_intervals))
    axs[1, 1].xaxis.set_minor_locator(AutoMinorLocator(5))
    axs[1, 1].yaxis.set_minor_locator(AutoMinorLocator(5))
    axs[1, 1].grid(which="major", color="#CCCCCC", linestyle="--")
    axs[1, 1].grid(which="minor", color="#CCCCCC", linestyle=":")
    axs[1, 1].ticklabel_format(style="sci", scilimits=(0, 0))

    axs[1, 1].plot(lims, lims, c=truth_color)  # Straight Line

    # axs[1, 1].scatter(train_predictions, train_labels, c=prediction_color)
    # axs[1, 1].grid(True, )
    # lims = [-.00001, .00013]
    # axs[1, 1].set_xlim(lims)
    # axs[1, 1].set_ylim(lims)

    # axs[1, 1].plot(lims, lims, c=truth_color) # Straight Line

    axs[0, 0].set(xlabel="gwdv [m/s^2]", ylabel="Train samples [.1 - 10 hPa / min]")
    axs[0, 1].set(xlabel="gwdv [m/s^2]", ylabel="Test samples [.1 - 10 hPa / min]")
    axs[1, 0].set(xlabel="gwdv [m/s^2]", ylabel="gwdv [m/s^2]")
    axs[1, 1].set(xlabel="gwdv [m/s^2]", ylabel="gwdv [m/s^2]")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

    plt.show()


def linear_p_vs_t():
    """
    Creates 2 plots
    Plot 1: 
        Red: num test labels vs test labels
        Blue: num prediction labels vs prediction labels 
    Plot 2: 
        num labels vs difference between test and prediction labels
    """
    nlabels = test_labels.shape[0]
    fig, ax = plt.subplots()
    ax.plot(test_labels, range(nlabels), "r", label="truth")
    ax.plot(test_predictions, range(nlabels), "b", label="predictions")
    legend = ax.legend(loc="upper center")
    plt.xlabel("gwdv m/s^2")
    plt.ylabel("Sample")
    plt.show()


def prediction_vs_truth():
    """
    Creates 1 plot
    x axis = true values
    y axis = predicted values
    """
    a = plt.axes(aspect="equal")
    plt.scatter(test_predictions, test_labels)
    plt.xlabel("True Values [m/s^2]")
    plt.ylabel("Predictions [m/s^2]")
    lims = [-9.4e-6, 0.00013]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    plt.show()


if __name__ == "__main__":
    grid_plot()
    # linear_p_vs_t()
    # prediction_vs_truth()
