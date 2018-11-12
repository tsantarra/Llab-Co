from math import sqrt

from utils.json_log_reader import *
from matplotlib.pyplot import *
from pandas import *
from analysis.plot_tools import *



def compare_success_rate(data_by_message):
    end_step_data = data_by_message['End Trial']

    grouped_data = end_step_data.groupby(paramlist, as_index=False)
    groups = Series(list(str(i) for i in grouped_data.groups.keys()))
    successes = Series([0]*len(groups))
    failures = Series([0]*len(groups))

    for i, (group, df) in enumerate(grouped_data):
        successes[i] = len(df[df['Reward'] == 100.0])
        failures[i] = len(df[df['Reward'] == 0.0])

    return stackedbarplot(x_data=groups,
                   y_data_list=[successes, failures],
                   y_data_names=['Success', 'Failure'])


def compare_success_rate_error_bars(data_by_message):
    end_step_data = data_by_message['End Trial']

    grouped_data = end_step_data.groupby(paramlist, as_index=False)
    groups = Series(list(str(i) for i in grouped_data.groups.keys()))
    success_rates = Series([0.0]*len(groups))
    conf_interval_sizes = Series([0.0] * len(groups))
    z = 1.96

    for i, (group, df) in enumerate(grouped_data):
        n = len(df)
        p = len(df[df['Reward'] == 100.0])/n
        success_rates[i] = p
        conf_interval_sizes[i] = z * sqrt(p * (1-p)/n)

    return error_bars(groups, success_rates, conf_interval_sizes)




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

    print('start')
    data_by_message = read_files_for_experiment('../' + data_dir, 32)

    #ax = compare_success_rate(data_by_message)
    compare_success_rate_error_bars(data_by_message)


    plt.show()
    print('Done.')