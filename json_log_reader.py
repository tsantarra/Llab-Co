
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
paramlist = ['plan_iterations',
             'comm_branch_factor',
             'comm_iterations',
             'scenario_id',
             'comm_cost',
             'process_no',
             'heuristic_id',
             'experience',
             'trials']


def read(file):
    with open(file, 'r') as json_file:
        param_line = loads(json_file.readline())
        params = {param: value for param, value in param_line.items()
                   if param not in {'timestamp', 'message', 'levelname'}}

        df = read_json(json_file, lines=True, convert_dates=True, convert_axes=True)
        df = df.assign(**params)
        df[['Trial']] = df[['Trial']].fillna(method='ffill')

    return params, df


def read_all_files(directory):
    dataset = {}
    for file in (f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.log')):
        params, df = read(join(directory, file))

        grouped_df = df.groupby(['levelname', 'message'])
        for group, groupdf in grouped_df:
            dataset[group] = groupdf if not group in dataset else concat([dataset[group], groupdf])

    return {group: groupdf.dropna(axis=1, how='all') for group, groupdf in dataset.items()}


def remove_nan_cols(dataframe):
    return dataframe.dropna(axis=1, how='all')


if __name__ == '__main__':
    dataset = read_all_files('login.osgconnect.net/out/')




