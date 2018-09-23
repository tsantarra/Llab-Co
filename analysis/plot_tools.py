from matplotlib import pyplot as plt


def confidence_interval_plot(x_data, y_data, sorted_x, low_CI, upper_CI, x_label, y_label, title):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the line, provide a label for the legend
    ax.plot(x_data, y_data, lw=1, color='#539caf', alpha=1, label='Fit')

    # Shade the confidence interval
    ax.fill_between(sorted_x, low_CI, upper_CI, color='#539caf', alpha=0.4, label='95% CI')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc='best')


























