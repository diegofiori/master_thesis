import numpy as np
import h5py
import os


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


def _get_phase_space(data, time_id, fields, tor_angle=None, remove_core=True):
    """ The method gives the discretized phase space of the fluid (plasma) in the simulation. It is possible both
    impose the coordinates axis to use or use all the information available. Since we only have the information
    at the grid points we call phase space the space having the physical quantities as axis. Each point corresponds
    to a grid point (x, y, psi) [or (psi, x, y) speaking with the formalism of the data]"""
    data = data['var3d']
    n_z, n_x, n_y = data[fields[0]][time_id].shape

    z = _recursive_extraction(data, 'coord3')[()].reshape((-1, 1, 1))
    x = _recursive_extraction(data, 'coord1')[()].reshape((1, -1, 1))
    y = _recursive_extraction(data, 'coord2')[()].reshape((1, 1, -1))

    z_bool_vec = np.array([True] * len(z))

    if tor_angle is not None:
        z_bool_vec = z[:, 0, 0] <= tor_angle
        n_z = np.sum(z_bool_vec)
        z = z[z_bool_vec]  # we take only a portion of the toroidal angle

    def flatten_grid(array, axis_rep):

        array = np.repeat(array, axis_rep[0], axis=0)
        array = np.repeat(array, axis_rep[1], axis=1)
        array = np.repeat(array, axis_rep[2], axis=2)

        return array.flatten()

    z_flat = np.expand_dims(flatten_grid(z, (1, n_x, n_y)), axis=1)
    y_flat = np.expand_dims(flatten_grid(y, (n_z, n_x, 1)), axis=1)
    x_flat = np.expand_dims(flatten_grid(x, (n_z, 1, n_y)), axis=1)
    list_of_vertical_features = [z_flat, x_flat, y_flat]
    for field in fields:
        list_of_vertical_features.append(np.expand_dims(data[field][time_id][()][z_bool_vec].flatten(), axis=1))

    features = np.concatenate(list_of_vertical_features, axis=1)

    if remove_core:
        xy_bool_vec = ((features[:, 1] > x.mean() + 0.05*(x.max()-x.min()))
                       | (features[:, 1] < x.mean() - 0.05*(x.max()-x.min()))
                       | (features[:, 2] > y.mean() + 0.05*(y.max()-y.min()))
                       | (features[:, 2] < y.mean() - 0.05*(y.max()-y.min())))
        features = features[xy_bool_vec]

    return features


def read_phase_space_file(path, time_step_list, fields, tor_angle=None, remove_core=True):
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
        field_data = _recursive_extraction(h5_file, fields[0])
        time_ids = list(set(time_step_list).intersection(set(field_data.keys())))
        data_list += [_get_phase_space(h5_file, t_id, fields, tor_angle, remove_core)
                      for t_id in time_ids]
    return np.array(data_list)


def read_average_field(path, field_name):
    h5_data = h5py.File(path, 'r')
    ph_q = _recursive_extraction(h5_data, field_name)[()]
    return ph_q


def get_all_time_ids(path):
    list_of_files = os.listdir(path)
    list_of_files = [name for name in list_of_files if ('result' in name)]
    list_of_files = sorted(list_of_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    time_ids = []
    for file in list_of_files:
        h5_file = _recursive_extraction(h5py.File(path + file, 'r')['data'], 'temperature')
        time_ids += [key for key in h5_file.keys() if key[0].isdigit()]
    return time_ids
