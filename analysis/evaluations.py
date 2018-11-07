from analysis.plot_tools import *
from utils.json_log_reader import paramlist

import matplotlib.pyplot as plt


def compare_success_rate(data_by_message):
    end_step_data = data_by_message['End Trial']
    heuristic_ids = np.unique(end_step_data['heuristic_id'])

    data = end_step_data.groupby(paramlist + ['Reward'], as_index=False).size().reset_index(name='counts')
    data['p'] = data['counts'] / 50
    data['std'] = np.sqrt(data['p'] * (1 - data['p']) / 50)
    print(data[data['Reward'] == 100.0])

    return stackedbarplot(x_data=heuristic_ids,
                   y_data_list=[data[data['Reward'] == 100.0]['counts'], data[data['Reward'] == 0.0]['counts']],
                   y_data_names=['Success', 'Failure'])


if __name__ == '__main__':
    from utils.json_log_reader import *
    from matplotlib.pyplot import *
    from pandas import *
    from analysis.plot_tools import *

    print('start')
    data_by_message = read_all_files('../' + data_dir)

    ax = compare_success_rate(data_by_message)  # todo - filter by process_no

    plt.show()
    print('Done.')