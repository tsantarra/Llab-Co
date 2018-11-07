
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

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pandas.io.json import read_json
    from pandas import concat


data_dir = 'login.osgconnect.net/out/'
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


def get_files_with_errors(directory):
    files_with_errors = []
    for file in (f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.log')):
        _, df = read(join(directory, file))

        if df['levelname'].isin(['ERROR']).any():
            files_with_errors.append(file)

    return files_with_errors


def remove_nan_cols(dataframe):
    return dataframe.dropna(axis=1, how='all')


if __name__ == '__main__':
    read_all_files(data_dir)