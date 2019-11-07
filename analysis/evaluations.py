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
        failures[i] = len(df[df['Reward'] <= 0.0])

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


def filter_by_filename(filename, experiment=None, scenario_id=None, heuristic_id=None, comm_branch_factor=None,
                       comm_iterations=None, comm_cost=None, plan_iterations=None, experience=None, alpha=None,
                       policy_cap=None):
    _, exp_no, scen, h, bf, ci, cc, pi, exp, _, a, pc, _, _ = filename.split('-')
    params = [exp_no, scen, h, bf, ci, cc, pi, exp, a, pc]
    filters = [experiment, scenario_id, heuristic_id, comm_branch_factor, comm_iterations, comm_cost, plan_iterations,
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


def output_table(group_cols, header_cols, baseline, data, alpha=0.05, caption='CAPTION'):
    """
    Example header:
        group_cols (separated) & trials & successes & p vs baseline & Avg Util & Util std & p vs baseline \\

    """
    assert len(group_cols) == len(header_cols)
    sep = '\t&\t'
    end = '\t\\\\'

    # Begin Table
    # print("\\begin{spacing}{1.0}\n\\begin{longtable}{" +
    #       'c'*(len(header_cols)+6) + "}\n\\caption{" +
    #       caption +
    #       "}\\label{tab:}\\\\\n\\toprule")
    print("\\begin{table}[!h] \n\\centering \n\\footnotesize \n\\caption[" + caption+"]{" + caption +
          "$\\mathbf{^*}$ denotes significant improvement over baseline.}%\\label{tab:} \n\\begin{tabular}{" +
          'c'*(len(header_cols)+6)+"}\n\\toprule")

    # Header
    header1 = ['']*(len(header_cols) + 3) + ['\\multicolumn{2}{c}{Reward}'] + ['']
    print(*header1, sep=sep, end='\t\\\\ \n')
    header2 = list(header_cols) + ['Trials', 'Successes', '$p_{success}$', 'Avg.', 'Std.', '$p_{util}$']
    print(*header2, sep=sep, end='\t\\\\ \\midrule\n')

    # Baseline
    baseline_successes = len(baseline[baseline['Reward'] > 0])
    baseline_failures = len(baseline[baseline['Reward'] <= 0])
    baseline_mean = baseline["Reward"].mean()
    baseline_std = baseline["Reward"].std()
    values = [baseline[col].iloc[0] for col in group_cols]
    values += [
        len(baseline),
        baseline_successes,
        '---',
        f'{baseline_mean:0.2f}',
        f'{baseline_std:0.2f}',
        '---',
    ]
    values[0] = 'No Comm.'
    print(*values, sep=sep, end='\t\\\\ \\hline\n')

    # subs
    subs = {
        0: '$\\sideset{}{_{E(\\action)}^{t}}\heuristic$',
        1: '$\\sideset{}{_{MAE}^{t}}\heuristic$',
        2: '$\\sideset{}{_{MSE}^{t}}\heuristic$',
        3: '$\\sideset{}{_{E(\\teammatepolicy)}^{t}}\heuristic$',
        4: '$\\sideset{}{_{\\voi}^{t}}\heuristic$',
        5: '$\\sideset{^w}{_{E(\\action)}^{t}}\heuristic$',
        6: '$\\sideset{^w}{_{MAE}^{t}}\heuristic$',
        7: '$\\sideset{^w}{_{MSE}^{t}}\heuristic$',
        8: '$\\sideset{^w}{_{E(\\teammatepolicy)}^{t}}\heuristic$',
        9: '$\\sideset{^w}{_{\\voi}^{t}}\heuristic$',
        10: '$\\sideset{}{_{E(\\teammatepolicy)}^{0}}\heuristic$',
        11: '$\\sideset{}{_{\\voi}^{0}}\heuristic$',
        12: '$\\sideset{}{_{U}^{t}}\heuristic$',
        13: '$\\sideset{^w}{_{}^{t}}\heuristic$',
        14: '$\\sideset{^w}{_{U}^{t}}\heuristic$',
        }
    subs = {
        0: 'Action Entropy',
        1: 'Mean Absolute Error',
        2: 'Mean Squared Error',
        3: '$\Delta$ Policy Entropy',
        4: 'Approx. Value of Info.',
        5: 'Weighted Action Entropy',
        6: 'Weighted Mean Abs. Error',
        7: 'Weighted Mean Sq. Error',
        8: 'Weighted $\Delta$ Policy Ent.',
        9: 'Weighted Approx. \\voi',
        10: 'Immediate Policy Ent.',
        11: 'Immediate Value of Info.',
        12: 'Uniform Random',
        13: 'State Likelihood',
        14: 'Weighted Uniform Random',
        }

    # Data
    for group, group_df in data.groupby(group_cols, as_index=False):
        # test means
        mean = group_df['Reward'].mean()
        mean_statistic, mean_p = ttest_ind(group_df['Reward'].values, baseline['Reward'].values, equal_var=False)

        # test success rate
        successes = len(group_df[group_df['Reward'] > 0.0])
        failures = len(group_df[group_df['Reward'] <= 0.0])
        binom_statistic, binom_p = fisher_exact([[
                successes,
                failures,
            ],
            [
                baseline_successes,
                baseline_failures,
            ],
        ], alternative='greater')

        try:
            values = list(group)
        except TypeError:
            values = [group]

        values += [
                len(group_df),
                successes,
                f'$\\mathbf{{{binom_p:0.3f}^*}}$' if (binom_p < alpha and successes > baseline_successes) else f'{binom_p:0.3f}',
                #f'\\textbf{{{binom_p:0.3f}}}' if binom_p < alpha else f'{binom_p:0.3f}',
                f'{group_df["Reward"].mean():0.2f}',
                f'{group_df["Reward"].std():0.2f}',
                f'$\\mathbf{{{mean_p:0.3f}^*}}$' if (mean_p < alpha and mean > baseline_mean) else f'{mean_p:0.3f}',
                #f'\\textbf{{{mean_p:0.3f}}}' if mean_p < alpha else f'{mean_p:0.3f}',
            ]

        # sub out heuristic id
        h = values[0]
        values[0] = subs[h]

        print(*values, sep=sep, end=end + ('\\midrule\n' if h in [4, 9, 11] else '\n'))


    # End Table
    #print("""\\bottomrule\n\\normalsize\n\\end{longtable}\n\\end{spacing}""", '\n\n')
    print("""\\bottomrule\n\\end{tabular}\n\\end{table}""", '\n\n')


def table_101():
    experiment = 101
    print(f'% {experiment}')
    groups = ['heuristic_id', ]
    group_labels = ['Heuristic', ]

    for it in [1, 10, 20]:
        for bf in [1, 3, 5]:
            base_filter = lambda f: filter_by_filename(f, experiment=experiment, comm_iterations=0)
            data_filter = lambda f: filter_by_filename(f, experiment=experiment, comm_iterations=it, comm_branch_factor=bf)

            baseline = read_files_for_experiment(data_dir, experiment, filter=base_filter)['End Trial']
            data = read_files_for_experiment(data_dir, experiment, filter=data_filter)['End Trial']

            output_table(groups, group_labels, baseline, data,
                         caption=f'Heuristics evaluation with communication branch factor of {bf} and {it} iteration(s) per search step.')


def table_102():
    experiment = 102
    print(f'% {experiment}')
    groups = ['heuristic_id', ]
    group_labels = ['Heuristic', ]

    for cost in [1, 5, 10, 99]:
        baseline = read_files_for_experiment(data_dir, experiment, filter=lambda f: filter_by_filename(f, experiment=experiment, comm_iterations=0))['End Trial']
        data = read_files_for_experiment(data_dir, experiment, filter=lambda f: filter_by_filename(f, experiment=experiment, comm_cost=cost))['End Trial']

        output_table(groups, group_labels, baseline, data,
                 caption=f'Agent coordinating with communication cost $C(\query)={cost}$.')


def table_103():
    experiment = 103
    print(f'% {experiment}')
    groups = ['heuristic_id', 'experience']
    group_labels = ['Heuristic', 'Experience']

    for exp in [0, 10, 100, 1000]:
        baseline = read_files_for_experiment(data_dir, experiment, filter=lambda f: filter_by_filename(f, experiment=experiment, comm_iterations=0, experience=exp))['End Trial']
        data = read_files_for_experiment(data_dir, experiment, filter=lambda f: filter_by_filename(f, experiment=experiment, experience=exp))['End Trial']

        output_table(groups, group_labels, baseline, data,
                     caption=f'Agent coordinating with {exp} episodes of past experience.')


def table_104():
    experiment = 104
    print(f'% {experiment}')
    groups = ['heuristic_id',]# 'experience', 'policy_cap']
    group_labels = ['Heuristic',]# 'Experience', 'Teammate Policies']

    for cap in [5, 25, 125]:
        for exp in [10, 100, 1000]:
            baseline = read_files_for_experiment(data_dir, experiment,
                                                 filter=lambda f: filter_by_filename(f,
                                                                                     experiment=experiment,
                                                                                     comm_iterations=0,
                                                                                     policy_cap=cap,
                                                                                     experience=exp))['End Trial']
            data = read_files_for_experiment(data_dir, experiment,
                                             filter=lambda f: filter_by_filename(f,
                                                                                 experiment=experiment,
                                                                                 comm_iterations=lambda ci: ci != 0,
                                                                                 policy_cap=cap,
                                                                                 experience=exp))['End Trial']

            output_table(groups, group_labels, baseline, data,
                         caption=f'Agent coordinating with {exp} experience with {cap} maximum unique teammate policies.')


def table_105():
    experiment = 105
    print(f'% {experiment}')
    groups = ['heuristic_id',]
    group_labels = ['Heuristic']

    for exp in [0, 10, 100, 1000]:
        baseline = read_files_for_experiment(data_dir, experiment, filter=lambda f: filter_by_filename(f, experiment=experiment, comm_iterations=0, experience=exp))['End Trial']
        data = read_files_for_experiment(data_dir, experiment, filter=lambda f: filter_by_filename(f, experiment=experiment, experience=exp))['End Trial']

        output_table(groups, group_labels, baseline, data,
                     caption=f'Agent coordinating with {exp} past episodes of experience.')


def print_section(title):
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\\clearpage\\section{' + title +
          '}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

    print('-'*100, '\n\n\n')

    print_section("Communication Search Parameters")
    table_101()  # search params

    print_section("Past Experience")
    table_103()  # exp

    print_section("Population Dynamics")
    table_104()  # pop cap

    print_section("Communication Cost")
    table_102()  # cost

    print_section("Domain Structure")
    table_105()   # domain structure



