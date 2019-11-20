import numpy as np
from giotto.pipeline import Pipeline
from sklearn.pipeline import make_pipeline, make_union

import os
import h5py


def _recursive_search(h5_data, index):
    flag = 0
    if isinstance(h5_data, list):
        return 0
    elif isinstance(h5_data, h5py._hl.group.Group):
        if index in h5_data.keys():
            return 1
        for key in h5_data.keys():
            flag += _recursive_search(h5_data[key], index)
    elif isinstance(h5_data, h5py._hl.dataset.Dataset):
        return 0
    return flag


def _recursive_extraction(h5_data, index):
    if isinstance(h5_data, h5py._hl.group.Group):
        if index in h5_data.keys():
            return h5_data[index]
        for key in h5_data.keys():
            temp = _recursive_extraction(h5_data[key], index)
            if temp is not None:
                return temp


def contains_id(file_name, id):
    file_h5 = h5py.File(file_name, 'r')
    bool_flag = _recursive_search(file_h5, id) > 0
    return bool_flag


def read_simulation_file(path, field_name, time_step_list):
    list_of_files = os.listdir(path)
    list_of_files = [name for name in list_of_files if ('result' in name)]
    list_of_files = sorted(list_of_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    list_of_wanted_files = []
    for file_name in list_of_files:
        for time_id in time_step_list:
            if contains_id(path + file_name, time_id):
                list_of_wanted_files.append(file_name)
    # no memory consuming
    list_of_wanted_files = sorted(list(set(list_of_wanted_files)), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    data_list = []
    for wanted_file_name in list_of_wanted_files:
        h5_file = h5py.File(path + wanted_file_name, 'r')['data']
        field_data = _recursive_extraction(h5_file, field_name)
        time_ids = list(set(time_step_list).intersection(set(field_data.keys())))
        data_list += [np.array(field_data[t_id][()]) for t_id in time_ids]

    return np.concatenate(data_list)


if __name__ == '__main__':
    import psutil
    print(psutil.virtual_memory())
    path = '/Users/diegofiori/Desktop/epfl/master_thesis/Reverse/'
    time_ids = ['001586', '001587', '001588', '001589', '001590']
    values = read_simulation_file(path, 'temperature', time_ids)
    print(values.shape)
    print(psutil.virtual_memory())
