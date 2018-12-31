from collections import Counter
from math import sqrt
from utils.json_log_reader import *
from matplotlib.pyplot import *
from pandas import Series
from analysis.plot_tools import *
from scipy.stats import mannwhitneyu, median_test, ttest_ind, fisher_exact
from scipy.stats.mstats import kruskalwallis

T_SCORE = 2.01  # 95% Confidence Interval


def compare_success_rate(data_by_message):
    end_step_data = data_by_message['End Trial']

    grouped_data = end_step_data.groupby(paramlist, as_index=False)
    groups = Series(list(str(i) for i in grouped_data.groups.keys()))
    successes = Series([0] * len(groups))
    failures = Series([0] * len(groups))

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

    for i, (group, df) in enumerate(grouped_data):
        n = len(df)
        p = len(df[df['Reward'] > 0.0]) / n
        success_rates.append(p)
        upper_conf_interval.append(min(T_SCORE * sqrt(p * (1 - p) / n), 1.0 - p))
        lower_conf_interval.append(min(T_SCORE * sqrt(p * (1 - p) / n), p))

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

        for g, df in secondary_grouping:
            n = len(df)
            p = len(df[df['Reward'] > 0.0]) / n
            success_rates.append(p)
            upper_conf_interval.append(min(T_SCORE * sqrt(p * (1 - p) / n), 1.0 - p))
            lower_conf_interval.append(min(T_SCORE * sqrt(p * (1 - p) / n), p))

        all_success_rates.append(success_rates)
        all_uppers.append(upper_conf_interval)
        all_lowers.append(lower_conf_interval)

    return grouped_error_bars(Series(list(range(len(all_success_rates[0])))),
                              [np.array(sr) for sr in all_success_rates],
                              [np.array([l, u]) for l, u in zip(all_lowers, all_uppers)],
                              **plot_kwargs)


def compare_end_utility_error_bars(grouped_data, baseline_data, **plot_kwargs):
    averages = []
    upper_conf_interval = []
    lower_conf_interval = []

    for i, (group, df) in enumerate(grouped_data):
        n = len(df)
        avg = df['Reward'].mean(axis=0)
        std = df['Reward'].std(axis=0)

        averages.append(avg)
        upper_conf_interval.append(min(T_SCORE * std / sqrt(n), 100 - avg))
        lower_conf_interval.append(min(T_SCORE * std / sqrt(n), avg))

    ax = error_bars(x_data=Series(list(str(i) for i in grouped_data.groups.keys())),
                    y_means=Series(averages),
                    y_err=np.array([lower_conf_interval, upper_conf_interval]),
                    **plot_kwargs)

    if baseline_data is None:
        return ax

    n = len(baseline_data)
    avg = baseline_data['Reward'].mean(axis=0)
    std = baseline_data['Reward'].std(axis=0)

    upper = min(T_SCORE * std / sqrt(n), 100 - avg)
    lower = min(T_SCORE * std / sqrt(n), avg)
    ax.axhspan(avg - lower, avg + upper, alpha=0.5, color='0.90')
    ax.axhline(avg, color='grey')
    return ax


def compare_end_utility_error_bars_grouped_treatments(grouped_data, treatment_var_list, baseline_data, **plot_kwargs):
    all_averages = []
    all_uppers = []
    all_lowers = []

    for group, data in grouped_data:
        secondary_grouping = data.groupby(treatment_var_list, as_index=False)

        averages = []
        upper_conf_interval = []
        lower_conf_interval = []

        for g, df in secondary_grouping:
            n = len(df)
            avg = df['Reward'].mean(axis=0)
            std = df['Reward'].std(axis=0)

            averages.append(avg)
            upper_conf_interval.append(min(T_SCORE * std / sqrt(n), 100 - avg))
            lower_conf_interval.append(min(T_SCORE * std / sqrt(n), avg))

        all_averages.append(averages)
        all_uppers.append(upper_conf_interval)
        all_lowers.append(lower_conf_interval)

    ax = grouped_error_bars(Series(list(range(len(all_averages[0])))),
                            [np.array(sr) for sr in all_averages],
                            [np.array([l, u]) for l, u in zip(all_lowers, all_uppers)],
                            **plot_kwargs)

    if baseline_data is None:
        return ax

    n = len(baseline_data)
    avg = baseline_data['Reward'].mean(axis=0)
    std = baseline_data['Reward'].std(axis=0)

    upper = min(T_SCORE * std / sqrt(n), 100 - avg)
    lower = min(T_SCORE * std / sqrt(n), avg)
    ax.axhspan(avg - lower, avg + upper, alpha=0.5, color='0.90')
    ax.axhline(avg, color='grey')
    return ax


def filter_by_filename(filename, process=None, scenario_id=None, heuristic_id=None, comm_branch_factor=None,
                       comm_iterations=None, comm_cost=None, plan_iterations=None, experience=None, alpha=None,
                       policy_cap=None):
    _, proc, scen, h, bf, ci, cc, pi, exp, _, a, pc, _, _ = filename.split('-')
    params = [proc, scen, h, bf, ci, cc, pi, exp, a, pc]
    filters = [process, scenario_id, heuristic_id, comm_branch_factor, comm_iterations, comm_cost, plan_iterations,
               experience, alpha, policy_cap]

    return all(filter is None or
               (filter(param) if callable(filter) else str(filter) == param)
               for filter, param in zip(filters, params))


def evaluate74():
    # 74 - varied heuristic [3, 4, 10, 11], branch factor [1, 2, 3, 5], iterations [1, 5, 10, 15, 20]
    # Want to show how varying branch factor and iterations changed success rates of each heuristic
    # baseline = read_files_for_experiment(data_dir, 74, filter=lambda f: filter_by_filename(f, comm_iterations=0))
    # for ci in [1, 5, 10, 15, 20]:
    #     data = read_files_for_experiment(data_dir, 74,
    #                                      filter=lambda f: filter_by_filename(f, comm_iterations=ci, ))
    #     grouped = data['End Trial'].sort_values(['comm_branch_factor'], ascending=True).groupby(
    #         ['comm_iterations', 'comm_branch_factor'], as_index=False)
    #     compare_end_utility_error_bars_grouped_treatments(grouped,
    #                                                       baseline_data=baseline['End Trial'],
    #                                                       treatment_var_list=['heuristic_id'],
    #                                                       tick_label=data['End Trial'].groupby(
    #                                                           ['heuristic_id']).groups.keys(),
    #                                                       title='Heuristic Success Rate in Cops and Robbers - CI=' + str(
    #                                                           ci),
    #                                                       x_label='Heuristic ID',
    #                                                       y_label='Success Rate',
    #                                                       y_range=(0, 100))
    #     plt.show()
    #
    # for bf in [1, 2, 3, 5]:
    #     data = read_files_for_experiment(data_dir, 74,
    #                                      filter=lambda f: filter_by_filename(f, comm_branch_factor=bf, ))
    #     grouped = data['End Trial'].sort_values(['comm_iterations'], ascending=True).groupby(
    #         ['comm_iterations', 'comm_branch_factor'], as_index=False)
    #     compare_end_utility_error_bars_grouped_treatments(grouped,
    #                                                       baseline_data=baseline['End Trial'],
    #                                                       treatment_var_list=['heuristic_id'],
    #                                                       tick_label=data['End Trial'].groupby(
    #                                                           ['heuristic_id']).groups.keys(),
    #                                                       title='Heuristic Success Rate in Cops and Robbers - BF=' + str(
    #                                                           bf),
    #                                                       x_label='Heuristic ID',
    #                                                       y_label='Success Rate',
    #                                                       y_range=(0, 100))
    #     plt.show()

    # 74 - varied heuristic [3, 4, 10, 11], branch factor [1, 2, 3, 5], iterations [1, 5, 10, 15, 20]
    # Want to show how varying branch factor and iterations changed success rates of each heuristic
    baseline = read_files_for_experiment(data_dir, 74, filter=lambda f: filter_by_filename(f, comm_iterations=0))
    for ci in [1, 5, 10, 15, 20]:
        for bf in [1, 2, 3, 5]:
            data = read_files_for_experiment(data_dir, 74,
                                             filter=lambda f: filter_by_filename(f,
                                                                                 comm_iterations=ci,
                                                                                 comm_branch_factor=bf))
            grouped = data['End Trial'].groupby(['heuristic_id'], as_index=False)
            compare_end_utility_error_bars(grouped,
                                           baseline_data=baseline['End Trial'],
                                           title=f'Heuristic Success Rate in Cops and Robbers - CI={ci} BF={bf}',
                                           x_label='Heuristic ID',
                                           y_label='Average Reward',
                                           y_range=(0, 100))
            plt.show()

    # Group all together - just to see which performs better than no comm.
    # baseline = read_files_for_experiment(data_dir, 74, filter=lambda f: filter_by_filename(f, comm_iterations=0))
    # data = read_files_for_experiment(data_dir, 74,
    #                                  filter=lambda f: filter_by_filename(f,))
    # grouped = data['End Trial'].groupby(['heuristic_id'], as_index=False)
    # compare_end_utility_error_bars(grouped,
    #                                baseline_data=baseline['End Trial'],
    #                                title=f'Heuristic Success Rate in Cops and Robbers',
    #                                x_label='Heuristic ID',
    #                                y_label='Average Reward',
    #                                y_range=(0, 100))
    # plt.show()


def evaluate75():
    for exp in [0, 10, 100, 1000]:
        baseline = read_files_for_experiment(data_dir, 75, filter=lambda f: filter_by_filename(f,
                                                                                               comm_iterations=0,
                                                                                               experience=exp))
        data = read_files_for_experiment(data_dir, 75,
                                         filter=lambda f: filter_by_filename(f,
                                                                             experience=exp))
        grouped = data['End Trial'].groupby(['heuristic_id'], as_index=False)

        print('Exp:', exp)
        mann_whitney_u(grouped, baseline['End Trial'])

        compare_end_utility_error_bars(grouped,
                                       baseline_data=baseline['End Trial'],
                                       title=f'Heuristic Success Rate in Cops and Robbers - EXP={exp}',
                                       x_label='Heuristic ID',
                                       y_label='Average Reward',
                                       y_range=(0, 100))
        plt.show()


def mann_whitney_u(grouped_data, baseline, alpha=0.05):
    for group, group_df in grouped_data:
        u, p = mannwhitneyu(group_df['Reward'], baseline['Reward'], alternative='greater')

        if p < alpha:
            print(group, group_df['Reward'].mean(), baseline['Reward'].mean(), u, p, '***')
        else:
            print(group, group_df['Reward'].mean(), baseline['Reward'].mean(), u, p)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

    trial = 105
    groups = ['heuristic_id', 'experience']  # 'comm_iterations', 'comm_branch_factor']

    tabs = Counter()
    baseline = read_files_for_experiment(data_dir, trial, filter=lambda f: filter_by_filename(f, comm_iterations=0))['End Trial']
    data = read_files_for_experiment(data_dir, trial, filter=lambda f: filter_by_filename(f,))

    grouped = data['End Trial'].groupby(groups, as_index=False)

    alpha = 0.05
    for group, group_df in grouped:
        #statistic, p = mannwhitneyu(group_df['Reward'].values,  baseline['Reward'].values, alternative='greater')
        #statistic, p = kruskalwallis(group_df['Reward'].values,  baseline['Reward'].values)
        #statistic, p, med, tbl = median_test(group_df['Reward'].values,  baseline['Reward'].values, ties='above')
        statistic, p = ttest_ind(group_df['Reward'].values,  baseline['Reward'].values, equal_var=False)

        if p < alpha:
            print(group, group_df['Reward'].mean(), group_df['Reward'].var(),  baseline['Reward'].mean(), baseline['Reward'].var(), statistic, p, '***')
            #tabs[group[0]] += 1
        else:
            print(group, group_df['Reward'].mean(), group_df['Reward'].var(),  baseline['Reward'].mean(), baseline['Reward'].var(), statistic, p, '')

        statistic, p = fisher_exact([   [
                                            len(group_df[group_df['Reward'] > 0]),
                                            len(group_df[group_df['Reward'] == 0])
                                        ],
                                        [
                                            len(baseline[baseline['Reward'] > 0]),
                                            len(baseline[baseline['Reward'] == 0])
                                        ],
                                    ], alternative='two-sided')

        if p < alpha:
            print(group, len(group_df[group_df['Reward'] > 0]), len(baseline[baseline['Reward'] > 0]), statistic, p, '***')
            tabs[group[0]] += 1
        else:
            print(group, len(group_df[group_df['Reward'] > 0]), len(baseline[baseline['Reward'] > 0]), statistic, p, '')


    print('Tally')
    print('\n'.join(str(item) for item in sorted(tabs.items())))
