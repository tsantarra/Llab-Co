from itertools import product
from os import listdir


def check_logs(vals, directory, log_found=False, min_count=1):
    filenames = listdir(directory)
    for permutation in product(*list(val[1] for val in vals)):
        file_start = 'data-' + '-'.join(map(str, permutation))
        permutation_count = sum(1 for filename in filenames if filename.startswith(file_start))

        if permutation_count < min_count:
            print(' '.join(map(str, permutation)) + ' (Cluster) $(Process)')
        elif log_found:
            print(' '.join(map(str, permutation)) + '\tFound ' + str(permutation_count))


if __name__ == '__main__':
    from notes.Experimental_Runs import data_config

    print('Missing: ')
    for experiment_number, parameters in data_config.items():
        if experiment_number < 70:
            continue

        check_logs(parameters, '../login.osgconnect.net/out/', log_found=False, min_count=2)


