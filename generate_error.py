import yaml

from joblib import Parallel, delayed
import numpy as np

from input_reader import get_all_time_ids

from run import process_images_data

FIELDS = ['strmf']


if __name__=='__main__':
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    print(cfg)
    simulation_path = '../Debug/'
    save_path = cfg['data_path']['save_path']
    n_time_steps = cfg['inputs']['n_time_steps']
    n_jobs = cfg['inputs']['n_jobs']
    resample_period = cfg['inputs']['resample_period']
    time_ids = get_all_time_ids(simulation_path)
    print(time_ids)
    features = Parallel(n_jobs=n_jobs)(delayed(process_images_data)(simulation_path, FIELDS[i],
                                                                    time_ids[j:min(j+n_time_steps, len(time_ids))],
                                                                    80)
                                       for j in range(0, len(time_ids), n_time_steps)
                                       for i in range(len(FIELDS)))
    # features = [np.array(features[i:i+len(FIELDS)]) for i in range(0, len(features)-len(FIELDS), len(FIELDS))]
    # features = np.concatenate(features, axis=1)
    print(features[0].shape)
