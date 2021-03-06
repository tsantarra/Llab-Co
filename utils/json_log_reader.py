
"""
How do I want to format my data for analysis?

(levelname, message) :  groupby (parameters -> data)

Access then will be:

    target_data = data[('INFO', 'EV')]
    target_data.sum()
    target_data.apply(func)  # func(group)

where
    def func(group):
        return pd.DataFrame({'original' : group,
                             'demeaned' : group - group.mean()})
"""
from json import loads
from os import listdir
from os.path import isfile, join
from csv import writer

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pandas.io.json import read_json
    from pandas import concat


data_dir = '../login.osgconnect.net/out/'
paramlist = ['process_no',
             'scenario_id',
             'heuristic_id',
             'comm_branch_factor',
             'comm_iterations',
             'comm_cost',
             'plan_iterations',
             'experience',
             'trials']


def read(file):
    with open(file, 'r') as json_file:
        param_line = loads(json_file.readline())
        params = {param: value for param, value in param_line.items()
                   if param not in {'timestamp', 'message', 'levelname'}}

        df = read_json(json_file, lines=True, convert_dates=True, convert_axes=True)
        df = df.assign(**params)

    return params, df


def read_all_files(directory):
    dataset = {}
    skipped_files = []
    for file in (f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.log')):
        params, df = read(join(directory, file))

        if df.empty:
            skipped_files.append(file)
            continue

        if df['levelname'].isin(['ERROR']).any():
            skipped_files.append(file)
            continue

        df[['Trial']] = df[['Trial']].fillna(method='ffill')
        for group, groupdf in df.groupby(['message']):
            dataset[group] = groupdf if not group in dataset else concat([dataset[group], groupdf], sort=False)

    if skipped_files:
        print('Files skipped:\n\t' + '\n\t'.join(skipped_files))

    return {group: groupdf.dropna(axis=1, how='all') for group, groupdf in dataset.items()}


def read_files_for_experiment(directory, experiment_no, filter=lambda f: True):
    dataset = {}
    skipped_files = []
    for file in (f for f in listdir(directory) if isfile(join(directory, f))
                                                  and f.startswith(f'data-{experiment_no}')
                                                  and filter(f)
                                                  and f.endswith('.log')):
        params, df = read(join(directory, file))

        if df.empty or df['levelname'].isin(['ERROR']).any():
            skipped_files.append(file)
            continue

        df[['Trial']] = df[['Trial']].fillna(method='ffill')
        for group, groupdf in df.groupby(['message']):
            dataset[group] = groupdf if not group in dataset else concat([dataset[group], groupdf], sort=False)

    if skipped_files:
        print('Files skipped:\n\t' + '\n\t'.join(skipped_files))

    return {group: groupdf.dropna(axis=1, how='all') for group, groupdf in dataset.items()}


def check_for_errors(directory, filter=lambda f: True, output_errors=True):
    files_with_errors = []
    for file in (f for f in listdir(directory)
                 if isfile(join(directory, f))
                  and filter(f)
                  and f.endswith('.log')):
        _, df = read(join(directory, file))

        if df['levelname'].isin(['ERROR']).any():
            files_with_errors.append(file)

            for row in df[df['levelname'] == 'ERROR']:
                print(row['message'])

    return files_with_errors


def remove_nan_cols(dataframe):
    return dataframe.dropna(axis=1, how='all')


def compile_to_csv(directory):
    skipped_files = []
    header = [
        'process_no',
        'scenario_id',
        'heuristic_id',
        'comm_branch_factor',
        'comm_iterations',
        'comm_cost',
        'plan_iterations',
        'experience',
        'trials',
        'alpha',
        'policy_cap',
        'successes',
        'avg',
        'std',
    ]
    data = [header]

    for file in (f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.log')):
        print(file)
        file_data, df = read(join(directory, file))

        if df.empty or df['levelname'].isin(['ERROR']).any():
            skipped_files.append(file)
            continue

        df[['Trial']] = df[['Trial']].fillna(method='ffill')
        groups = df.groupby(['message'])
        ends = groups.get_group('End Trial')

        file_data['successes'] = len(ends[ends['Reward'] > 0.0])
        file_data['avg'] = ends['Reward'].mean(axis=0)
        file_data['std'] = ends['Reward'].std(axis=0)

        data.append([file_data[key] for key in header])

    if skipped_files:
        print('Files skipped:\n\t' + '\n\t'.join(skipped_files))

    with open('../data_aggregates.csv', 'w', newline='') as csvfile:
        csv_writer = writer(csvfile)
        csv_writer.writerows(data)


if __name__ == '__main__':
    compile_to_csv(data_dir)