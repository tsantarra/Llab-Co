#!/usr/bin/python
import sys
import csv

if __name__ == '__main__':
    assert len(sys.argv) == 3, 'INCORRECT ARGUMENTS: ' + str(sys.argv)

    _, arg1, arg2 = sys.argv

    with open('out/' + arg1 + '-' + arg2 +'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(sys.argv)

        # for testing virtual env
        from logmatic import jsonlogger
        writer.writerow([jsonlogger.time])
