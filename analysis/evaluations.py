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


def compare_success_rate_error_bars(grouped_data, **plot_kwargs):
    success_rates = []
    upper_conf_interval = []
    lower_conf_interval = []
    z = 1.96                       # 95% Confidence Interval

    for i, (group, df) in enumerate(grouped_data):
        n = len(df)
        p = len(df[df['Reward'] == 100.0])/n
        success_rates.append(p)
        upper_conf_interval.append(min(z * sqrt(p * (1-p)/n), 1.0 - p))
        lower_conf_interval.append(min(z * sqrt(p * (1-p)/n), p ))

    return error_bars(x_data=Series(list(str(i) for i in grouped_data.groups.keys())),
                      y_means=Series(success_rates),
                      y_err=np.array([lower_conf_interval, upper_conf_interval]),
                      **plot_kwargs)


def compare_success_rate_error_bars_grouped_treatments(grouped_data, treatment_var_list, **plot_kwargs):
    all_success_rates = []
    all_uppers = []
    all_lowers = []

    for group, data in grouped_data:
        secondary_grouping = data.groupby(treatment_var_list, as_index=False)

        success_rates = []
        upper_conf_interval = []
        lower_conf_interval = []
        z = 1.96                       # 95% Confidence Interval

        for g, df in secondary_grouping:
            n = len(df)
            p = len(df[df['Reward'] == 100.0])/n
            success_rates.append(p)
            upper_conf_interval.append(min(z * sqrt(p * (1-p)/n), 1.0 - p))
            lower_conf_interval.append(min(z * sqrt(p * (1-p)/n), p ))

        all_success_rates.append(success_rates)
        all_uppers.append(upper_conf_interval)
        all_lowers.append(lower_conf_interval)

    return grouped_error_bars(Series(list(range(len(all_success_rates[0])))),
                              [np.array(sr) for sr in all_success_rates], #Series(all_success_rates),
                              [np.array([l,u]) for l,u in zip(all_lowers, all_uppers)],
                              **plot_kwargs)



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

    if False:
        # 32 - Pt 1
        data = read_files_for_experiment(data_dir, 32, filter=lambda f: f.endswith('0.log'))
        compare_success_rate_error_bars(data['End Trial'].groupby(['heuristic_id'], as_index=False),
                                        title='Heuristic Success Rate in Cops and Robbers - Alpha=0',
                                        x_label='Heuristic ID',
                                        y_label='Average Success',
                                        y_range=(0, 1))
        plt.show()

        # 32 - Pt 2
        data = read_files_for_experiment(data_dir, 32, filter=lambda f: f.endswith('1.log'))
        compare_success_rate_error_bars(data['End Trial'].groupby(['heuristic_id'], as_index=False),
                                        title='Heuristic Success Rate in Cops and Robbers - Alpha=1',
                                        x_label='Heuristic ID',
                                        y_label='Average Success',
                                        y_range=(0, 1))
        plt.show()

        # 34 - A handful of heuristics evaluated with different branch factors
        #       (show each heuristic alone with 4 grouped treatments)
        data = read_files_for_experiment(data_dir, 34)
        grouped = data['End Trial'].groupby(['comm_branch_factor'], as_index=False)
        compare_success_rate_error_bars_grouped_treatments(grouped,
                                                           treatment_var_list=paramlist,
                                                           tick_label=data['End Trial'].groupby(['heuristic_id']).groups.keys(),
                                                            title='Heuristic Success Rate in Cops and Robbers - Alpha=1',
                                                            x_label='Heuristic ID',
                                                            y_label='Average Success',
                                                            y_range=(0, 1))
        plt.show()


    # 53 - Rerun all heuristics, but with only 1 comm planning iteration
    data = read_files_for_experiment(data_dir, 32, filter=lambda f: f.endswith('1.log'))
    compare_success_rate_error_bars(data['End Trial'].groupby(['heuristic_id'], as_index=False),
                                    title='Heuristic Success Rate in Cops and Robbers - Alpha=1',
                                    x_label='Heuristic ID',
                                    y_label='Average Success',
                                    y_range=(0, 1))
    plt.show()






