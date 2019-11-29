import numpy as np
from giotto.diagrams import Scaler, PersistenceEntropy, Amplitude
from giotto.homology import VietorisRipsPersistence
from giotto.pipeline import make_pipeline
from sklearn.pipeline import make_union, FeatureUnion
from joblib import Parallel, delayed

from resampler import Grouper, Degrouper
from diagram_derivatives import MultiDiagramsDerivative
from utils import write_pickle

import os
import h5py


METRIC_LIST = [
    {'metric': 'bottleneck', 'metric_params': {'p': np.inf}},
    {'metric': 'wasserstein', 'metric_params': {'p': 1}},
    {'metric': 'wasserstein', 'metric_params': {'p': 2}},
    {'metric': 'landscape', 'metric_params': {'p': 1, 'n_layers': 1, 'n_values': 100}},
    {'metric': 'landscape', 'metric_params': {'p': 1, 'n_layers': 2, 'n_values': 100}},
    {'metric': 'landscape', 'metric_params': {'p': 2, 'n_layers': 1, 'n_values': 100}},
    {'metric': 'landscape', 'metric_params': {'p': 2, 'n_layers': 2, 'n_values': 100}},
    {'metric': 'betti', 'metric_params': {'p': 1, 'n_values': 100}},
    {'metric': 'betti', 'metric_params': {'p': 2, 'n_values': 100}},
    {'metric': 'heat', 'metric_params': {'p': 1, 'sigma': 1.6, 'n_values': 100}},
    {'metric': 'heat', 'metric_params': {'p': 1, 'sigma': 3.2, 'n_values': 100}},
    {'metric': 'heat', 'metric_params': {'p': 2, 'sigma': 1.6, 'n_values': 100}},
    {'metric': 'heat', 'metric_params': {'p': 2, 'sigma': 3.2, 'n_values': 100}}
]

DERIVATIVE_METRIC_LIST = [
    {'metric': 'bottleneck', 'metric_params': {'p': np.inf}},
    {'metric': 'wasserstein', 'metric_params': {'p': 1}},
    {'metric': 'wasserstein', 'metric_params': {'p': 2}}
]
FIELDS = ['theta', 'temperature', 'temperaturi', 'strmf', 'vpari']
GLOBAL_FIELDS = ['globtheta']


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


def get_all_time_ids(path):
    list_of_files = os.listdir(path)
    list_of_files = [name for name in list_of_files if ('result' in name)]
    list_of_files = sorted(list_of_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    time_ids = []
    for file in list_of_files:
        h5_file = _recursive_extraction(h5py.File(path + file, 'r')['data'], 'temperature')
        time_ids += [key for key in h5_file.keys() if key[0].isdigit()]
    return time_ids


def build_the_space_pipeline(space_period, n_jobs=None):
    rips = VietorisRipsPersistence(homology_dimensions=[0, 1])
    scaler = Scaler(metric='bottleneck')
    initial_pipeline_elements = [rips,  scaler]
    entropy_step = [initial_pipeline_elements + [PersistenceEntropy()]]
    amplitude_steps = [initial_pipeline_elements+[Amplitude(**metric, order=None)] for metric in METRIC_LIST]
    amplitudes_entropy_steps = entropy_step + amplitude_steps
    # amplitudes_pipeline = make_union(*[make_pipeline(*steps) for steps in amplitudes_entropy_steps], n_jobs=n_jobs)
    # now we build the derivative pipeline
    grouper = Grouper(period=space_period)
    degrouper = Degrouper()
    derivative_steps = [initial_pipeline_elements +
                        [grouper, MultiDiagramsDerivative(**metric, periodic=True, order=None), degrouper]
                        for metric in DERIVATIVE_METRIC_LIST]
    all_steps = amplitudes_entropy_steps + derivative_steps
    total_pipeline = make_union(*[make_pipeline(*steps) for steps in all_steps], n_jobs=n_jobs)

    return total_pipeline


def process_images_data(path, field_name, time_step_list, len_z):
    data_loaded = read_simulation_file(path, field_name, time_step_list)
    pipeline = build_the_space_pipeline(space_period=len_z)
    features = pipeline.fit_transform(data_loaded)
    return features

def process_physics_data(path, field_name):
    h5_data = h5py.File(path, 'r')
    ph_q = _recursive_extraction(h5_data, field_name)
    return ph_q


if __name__ == '__main__':
    simulation_path = '/Users/diegofiori/Desktop/epfl/master_thesis/Reverse/'
    save_path = '/Users/diegofiori/Desktop/epfl/master_thesis/results/'
    n_time_steps = 6
    n_jobs = -1
    time_ids = get_all_time_ids(simulation_path)[:10]
    if not os.path.isfile(save_path + 'slices_top_features.pickle'):
        features = Parallel(n_jobs=n_jobs)(delayed(process_images_data)(simulation_path, FIELDS[i],
                                                                        time_ids[j:min(j+n_time_steps, len(time_ids))], 80)
                                           for j in range(0, len(time_ids), n_time_steps)
                                           for i in range(len(FIELDS)))
        features = [np.array(features[i:i+len(FIELDS)]) for i in range(0, len(features)-len(FIELDS), len(FIELDS))]
        features = np.concatenate(features, axis=1)
        write_pickle(save_path+'slices_top_features.pickle', features)

    if not os.path.isfile(save_path+'physical_features.pickle'):
        file_names = [name for name in os.listdir(simulation_path) if ('result' in name)]
        file_names = sorted(file_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        ph_features = Parallel(n_jobs=n_jobs)(delayed(process_physics_data(simulation_path+file_names[i]))
                                              for i in range(len(file_names)))
        ph_features = np.array(ph_features)
        write_pickle(save_path+'physical_features.pickle', ph_features)
    
    # print('phase space')
    # if not os.path.isfile(save_path+'phase_space_top_features.pickle'):
    #     phase_space_top = Parallel(n_jobs=-1)(delayed(process_result_phase_space)(dir_path[t], dir_path[t+1])
    #                                           for t in range(len(dir_path) - 1))
    #
    #     phase_space_top = np.concatenate(phase_space_top)
    #
    #     write_pickle(path=save_path+'phase_space_top_features.pickle', array=phase_space_top)
    #
    # print('extract physics')
    # if not os.path.isfile(save_path+'physical_features.pickle'):
    #     physical_qs = Parallel(n_jobs=-1)(delayed(process_result_physical_quantities)(dir_path[t])
    #                                       for t in range(len(dir_path) - 1))
    #     physical_qs = pd.concat(physical_qs)
    #     physical_qs.to_pickle(save_path+'physical_features.pickle')
