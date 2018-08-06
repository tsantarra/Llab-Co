from json import loads
from pandas import DataFrame


def read(file):
    with open(file, 'r') as json_file:
        records = [loads(line) for line in json_file]

    for record in records:
        print(DataFrame.from_dict({k: [v] for k,v in record.items()}))
        break
        # pull out the keys per record type (different keys for different info) - new DataFrame for each
        # values are just rows (or columns?) of the keyed array

    #print('\n'.join(str(sorted(record.items())) for record in records))


if __name__ == '__main__':
    read('login.osgconnect.net/out/data-1.log')
