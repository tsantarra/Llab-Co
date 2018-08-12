from pandas.io.json import read_json
from json import loads
from os import listdir
from os.path import isfile, join


data_file = 'login.osgconnect.net/out/data-1.log'


def read(file):
    with open(file, 'r') as json_file:
        param_line = loads(json_file.readline())
        df = read_json(json_file, lines=True)

        params = {param: value for param, value in param_line.items()
                   if param not in {'timestamp', 'message', 'levelname'}}
        df = df.assign(**params)

    return params, df


def read_all_files(directory):
    dataset = {}
    for file in (f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.log')):
        params, df = read(join(directory, file))
        dataset[params] = df

        df2 = df.sort_values(by=[''])

    return dataset


def print_five(df):
    print(df[['message']][0:5])


if __name__ == '__main__':
    dataset = read_all_files('login.osgconnect.net/out/')



