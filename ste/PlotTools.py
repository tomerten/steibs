import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy import stats


def distributionplot(arr):
    label = ["hor", "ver", "long"]  # List of labels for categories
    cl = ["b", "r", "y"]  # List of colours for categories
    categories = len(label)
    sample_size = arr.shape[0]  # Number of samples in each category

    # Create numpy arrays for dummy x and y data:
    x = np.zeros(shape=(categories, sample_size))
    y = np.zeros(shape=(categories, sample_size))

    # Generate random data for each categorical variable:
    for n in range(0, categories):
        x[n, :] = arr[:, n * 2]
        y[n, :] = arr[:, 1 + 2 * n]

    # Set up 8 subplots as axis objects using GridSpec:
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 3, 1, 3], height_ratios=[3, 1])
    # Add space between scatter plot and KDE plots to accommodate axis labels:
    gs.update(hspace=0.3, wspace=0.3)

    # Set background canvas colour to White instead of grey default
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("white")

    # Transverse plots
    ax = plt.subplot(gs[0, 1])  # Instantiate scatter plot area and axis range
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    axl = plt.subplot(gs[0, 0], sharey=ax)  # Instantiate left KDE plot area
    axl.get_xaxis().set_visible(False)  # Hide tick marks and spines
    axl.get_yaxis().set_visible(False)
    axl.spines["right"].set_visible(False)
    axl.spines["top"].set_visible(False)
    axl.spines["bottom"].set_visible(False)

    axb = plt.subplot(gs[1, 1], sharex=ax)  # Instantiate bottom KDE plot area
    axb.get_xaxis().set_visible(False)  # Hide tick marks and spines
    axb.get_yaxis().set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axb.spines["left"].set_visible(False)

    axc = plt.subplot(gs[1, 0])  # Instantiate legend plot area
    axc.axis("off")  # Hide tick marks and spines

    # Longitudinal plots
    ax2 = plt.subplot(gs[0, 3])  # Instantiate scatter plot area and axis range
    ax2.set_xlim(x[2].min(), x[2].max())
    ax2.set_ylim(y[2].min(), y[2].max())
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    axl2 = plt.subplot(gs[0, 2], sharey=ax2)  # Instantiate left KDE plot area
    axl2.get_xaxis().set_visible(False)  # Hide tick marks and spines
    axl2.get_yaxis().set_visible(False)
    axl2.spines["right"].set_visible(False)
    axl2.spines["top"].set_visible(False)
    axl2.spines["bottom"].set_visible(False)

    axb2 = plt.subplot(gs[1, 3], sharex=ax2)  # Instantiate bottom KDE plot area
    axb2.get_xaxis().set_visible(False)  # Hide tick marks and spines
    axb2.get_yaxis().set_visible(False)
    axb2.spines["right"].set_visible(False)
    axb2.spines["top"].set_visible(False)
    axb2.spines["left"].set_visible(False)

    axc2 = plt.subplot(gs[1, 2])  # Instantiate legend plot area
    axc2.axis("off")  # Hide tick marks and spines

    # Plot data for each categorical variable as scatter and marginal KDE plots:
    for n in range(0, categories - 1):
        ax.scatter(x[n], y[n], color="none", label=label[n], s=1, edgecolor=cl[n])

        kde = stats.gaussian_kde(x[n, :])
        xx = np.linspace(x.min(), x.max(), 1000)
        axb.plot(xx, kde(xx), color=cl[n])

        kde = stats.gaussian_kde(y[n, :])
        yy = np.linspace(y.min(), y.max(), 1000)
        axl.plot(kde(yy), yy, color=cl[n])

    n = 2
    ax2.scatter(x[n], y[n], color="none", label=label[n], s=1, edgecolor=cl[n])

    kde = stats.gaussian_kde(x[n, :])
    xx = np.linspace(x[n].min(), x[n].max(), 1000)
    axb2.plot(xx, kde(xx), color=cl[n])

    kde = stats.gaussian_kde(y[n, :])
    yy = np.linspace(y[n].min(), y[n].max(), 1000)
    axl2.plot(kde(yy), yy, color=cl[n])

    # Copy legend object from scatter plot to lower left subplot and display:
    # NB 'scatterpoints = 1' customises legend box to show only 1 handle (icon) per label
    handles, labels = ax.get_legend_handles_labels()
    axc.legend(handles, labels, scatterpoints=1, loc="center", fontsize=12)

    handles, labels = ax2.get_legend_handles_labels()
    axc2.legend(handles, labels, scatterpoints=1, loc="center", fontsize=12)

    plt.show()
