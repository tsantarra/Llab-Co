
from json import loads
from collections import Counter
from utils.json_log_reader import paramlist, read_all_files, data_dir

import csv

def process_queries(data_by_message, out_dir):
    query_data = data_by_message['Query Step']

    group_data = {}
    max_dim = 0

    for group, group_df in query_data.groupby(paramlist):
        flattened = Counter( (value if type(value) is int else tuple(value)
                              for query in group_df['Query']
                              for key, value in loads(query).items() if not key.startswith('Rob') and key != 'A') )

        round_data = Counter(dict(item for item in flattened.items() if type(item[0]) is int))
        loc_data = Counter(dict(item for item in flattened.items() if type(item[0]) is tuple))

        group_data[group] = (loc_data, round_data)
        max_dim = max(max_dim, max((key[0] for key in loc_data)), max((key[1] for key in loc_data)))


    with open(out_dir + '/results.csv', 'w', newline='') as f:
        writer = csv.writer(f,)

        skip = len(paramlist) + 1

        for group, (loc_data, round_data) in group_data.items():

            writer.writerow([])
            writer.writerow(list(group))

            for row in range(max_dim+1):
                writer.writerow([None] * skip + list(loc_data[(row, col)] for col in range(max_dim+1)) + [None, round_data[row+1]])

            writer.writerow([])




if __name__ == '__main__':
    dir = './../' + data_dir
    process_queries(read_all_files(dir), dir)

