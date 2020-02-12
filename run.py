import numpy as np
from joblib import Parallel, delayed
import yaml

from pipeline_builder import build_the_space_pipeline, build_phase_space_pipeline
from utils import write_pickle
from input_reader import read_simulation_file, read_phase_space_file, get_all_time_ids, read_average_field

import os


FIELDS = ['theta', 'temperature', 'temperaturi', 'strmf', 'vpari']
GLOBAL_FIELDS = ['globtheta']
NEED_EXP = ['theta', 'temperature', 'temperaturi']


def process_images_data(path, field_name, time_step_list, len_z, comp_rem=0):
    data_loaded = read_simulation_file(path, field_name, time_step_list)
    pipeline = build_the_space_pipeline(space_period=len_z,
                                        remove_n_comp=comp_rem)
    if field_name in NEED_EXP:
        data_loaded = np.exp(data_loaded)
    features = pipeline.fit_transform(data_loaded)
    return features


def process_phase_space_data(path, time_step_list, previous_time_step, resample_period):
    time_step_list = [previous_time_step] + time_step_list
    data_loaded = read_phase_space_file(path, time_step_list, FIELDS, tor_angle=None, remove_core=True)
    pipeline = build_phase_space_pipeline(resample_period)
    features = pipeline.fit_transform(data_loaded)
    return features


def process_physics_data(path, field_name):
    return read_average_field(path, field_name)


if __name__ == '__main__':
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    print(cfg)
    simulation_path = cfg['data_path']['simulation_path']
    save_path = cfg['data_path']['save_path']
    n_time_steps = cfg['inputs']['n_time_steps']
    n_jobs = cfg['inputs']['n_jobs']
    resample_period = cfg['inputs']['resample_period']
    comp_rem = cfg['inputs']['nb_components_to_remove']
    time_ids = get_all_time_ids(simulation_path)[-100:]
    if not os.path.isfile(save_path + 'slices_top_features.pickle'):
        features = Parallel(n_jobs=n_jobs)(delayed(process_images_data)(simulation_path, FIELDS[i],
                                                                        time_ids[j:min(j+n_time_steps, len(time_ids))],
                                                                        80, comp_rem)
                                           for j in range(0, len(time_ids), n_time_steps)
                                           for i in range(len(FIELDS)))
        features = [np.array(features[i:i+len(FIELDS)]) for i in range(0, len(features)-len(FIELDS), len(FIELDS))]
        features = np.concatenate(features, axis=1)
        write_pickle(save_path+'slices_top_features.pickle', features)
    if not os.path.isfile(save_path + 'phase_space_top_features.pickle'):
        ps_features = Parallel(n_jobs=n_jobs)(delayed(process_phase_space_data)
                                              (simulation_path, time_ids[j:min(j+n_time_steps, len(time_ids))],
                                               time_ids[j-1], resample_period)
                                              for j in range(1, len(time_ids), n_time_steps))
        write_pickle(save_path + 'phase_space_top_features.pickle', ps_features)

    if not os.path.isfile(save_path+'physical_features.pickle'):
        file_names = [name for name in os.listdir(simulation_path) if ('result' in name)]
        file_names = sorted(file_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        ph_features = Parallel(n_jobs=n_jobs)(delayed(process_physics_data)(simulation_path+file_names[i],
                                                                            GLOBAL_FIELDS[j])
                                              for j in range(len(GLOBAL_FIELDS))
                                              for i in range(len(file_names)))
        ph_features = np.array(ph_features)
        write_pickle(save_path+'physical_features.pickle', ph_features)


