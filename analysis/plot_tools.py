"""
Examples from:
    https://www.datascience.com/blog/learn-data-science-intro-to-data-visualization-in-matplotlib

Useful references:
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
"""
from matplotlib import pyplot as plt
import numpy as np


def lineplot(x_data, y_data, x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw=2, color='#539caf', alpha=1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def lineplot2y(x_data, x_label, y1_data, y1_color, y1_label, y2_data, y2_color, y2_label, title):
    # Each variable will actually have its own plot object but they
    # will be displayed in just one plot
    # Create the first plot object and draw the line
    _, ax1 = plt.subplots()
    ax1.plot(x_data, y1_data, color=y1_color)
    # Label axes
    ax1.set_ylabel(y1_label, color=y1_color)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(x_data, y2_data, color=y2_color)
    ax2.set_ylabel(y2_label, color=y2_color)
    # Show right frame line
    ax2.spines['right'].set_visible(True)


def confidence_interval_plot(x_data, y_data, sorted_x, low_ci, upper_ci, x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the line, provide a label for the legend
    ax.plot(x_data, y_data, lw=1, color='#539caf', alpha=1, label='Fit')

    # Shade the confidence interval
    ax.fill_between(sorted_x, low_ci, upper_ci, color='#539caf', alpha=0.4, label='95% CI')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc='best')


def scatterplot(x_data, y_data, x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s=30, color='#539caf', alpha=0.75)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def histogram(data, x_label, y_label, title):
    _, ax = plt.subplots()
    ax.hist(data, color='#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


def overlaid_histogram(data1, data1_name, data1_color, data2, data2_name, data2_color, x_label, y_label, title):
    # Set the bounds for the bins so that the two distributions are
    # fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins
    bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins=bins, color=data1_color, alpha=1, label=data1_name)
    ax.hist(data2, bins=bins, color=data2_color, alpha=0.75, label=data2_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='best')


def barplot(x_data, y_data, error_data, x_label, y_label, title):
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color='#539caf', align='center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
    ax.errorbar(x_data, y_data, yerr=error_data, color='#297083', ls='none', lw=2, capthick=2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


def stackedbarplot(x_data, y_data_list, y_data_names, colors=['#539caf', '#7663b0'], x_label='', y_label='', title=''):
    _, ax = plt.subplots()
    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i], color=colors[i], align='center', label=y_data_names[i])
        else:
            # For each category after the first, the bottom of the
            # bar will be the top of the last category
            ax.bar(x_data, y_data_list[i], color=colors[i], bottom=y_data_list[i - 1], align='center',
                   label=y_data_names[i])

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='upper right')

    return ax


def groupedbarplot(x_data, y_data_list, y_data_names, colors, x_label, y_label, title):
    _, ax = plt.subplots()
    # Total width for all bars at one x location
    total_width = 0.8
    # Width of each individual bar
    ind_width = total_width / len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width / 2), total_width / 2, ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ax.bar(x_data + alteration[i], y_data_list[i], color=colors[i], label=y_data_names[i], width=ind_width)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='upper right')


def boxplot(x_data, y_data, base_color='#539caf', median_color='#539caf', x_label='', y_label='', title='', **kwargs):
    _, ax = plt.subplots()

    # Draw boxplots, specifying desired style
    ax.boxplot(y_data,
               # patch_artist must be True to control box fill
               patch_artist=True,
               # Properties of median line
               medianprops={'color': median_color},
               # Properties of box
               boxprops={'color': base_color, 'facecolor': base_color},
               # Properties of whiskers
               whiskerprops={'color': base_color},
               # Properties of whisker caps
               capprops={'color': base_color},
               **kwargs)

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


def error_bars(x_data, y_means, y_err, x_label='', y_label='', title='', y_range=None, **kwargs):
    _, ax = plt.subplots()

    if y_range:
        ax.set_ylim(*y_range)

    ax.errorbar(x_data, y_means, yerr=y_err, fmt='ks:', ls='none', capsize=2, **kwargs)
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    return ax


def grouped_error_bars(x_data, y_means_list, y_err_list, x_label='', y_label='', title='', tick_label=None, y_range=None, **kwargs):
    _, ax = plt.subplots()
    ax.set_xticks(list(range(len(y_means_list))))

    if y_range:
        ax.set_ylim(*y_range)

    # Total width for all bars at one x location
    total_width = 0.5
    # Width of each individual bar
    ind_width = total_width / len(y_means_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width / 2), total_width / 2, ind_width)

    # Draw bars, one category at a time
    for i in range(len(y_means_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ax.errorbar(x_data + alteration[i], y_means_list[i], yerr=y_err_list[i], fmt='ks:', ls='none', capsize=2, **kwargs)
        #ax.errorbar(x_data + alteration[i], y_data_list[i], color=colors[i], label=y_data_names[i], width=ind_width)

    if tick_label is not None:
        ax.set_xticklabels(tick_label)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='upper right')

    return ax


if __name__== '__main__':
    import numpy as np
    import pylab as pl

    # define datasets
    parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    mean_1 = [10.1, 12.1, 13.6, 14.5, 18.8, 11.8, 28.5]
    std_1 = [2.6, 5.7, 4.3, 8.5, 11.8, 5.3, 2.5]

    mean_2 = [10.1, 12.1, 13.6, 14.5, 18.8, 11.8, 28.5]
    std_2 = [2.6, 5.7, 4.3, 8.5, 11.8, 5.3, 2.5]

    mean_3 = [10.1, 12.1, 13.6, 14.5, 18.8, 11.8, 28.5]
    std_3 = [2.6, 5.7, 4.3, 8.5, 11.8, 5.3, 2.5]

    grouped_error_bars(parameters, [mean_1, mean_2, mean_3], [std_1, std_2, std_3], ['#539caf', '#539caf', '#539caf'])

    pl.show()
