import giotto as o
from giotto.pipeline import Pipeline
from giotto.diagrams import Amplitude, Scaler, PersistenceEntropy
from giotto.homology import VietorisRipsPersistence, CubicalPersistence
import numpy as np
import pandas as pd

from input_reader import ImageReader, PhaseSpaceReader
from resampler import Grouper, ShiftResampler, Resampler
from utils import read_pickle, write_pickle
from simulation import Simulation
from joblib.parallel import Parallel, delayed
from diagram_derivatives import MultiDiagramsDerivative, DiagramDerivative
import os


SIMULATION_PATH = '/Users/diegofiori/Desktop/epfl/master_thesis/Reverse/'
SAVE_PATH = '/Users/diegofiori/Desktop/epfl/master_thesis/results/'
# METRICS = ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']
METRICS = ['landscape']


def process_result_image(result_name_pres, result_name_fut):
    image_reader = ImageReader()
    images_pres = image_reader.read(SIMULATION_PATH+result_name_pres)
    images = images_pres
    z_fut = 0
    if result_name_fut is not None:
        images_fut = image_reader.read(SIMULATION_PATH+result_name_fut)
        images = np.concatenate([images_pres, images_fut[:image_reader.structure_['dim_z']]])
        z_fut = image_reader.structure_['dim_z']
    diagrams = CubicalPersistence().fit_transform(images)

    fake_y = np.zeros(len(diagrams))
    list_of_pipeline_amplitude = [Pipeline([('rescale_diagrams', Scaler(metric=metric)),
                                            ('compute_amplitude', Amplitude(metric=metric)),
                                            ('group_spatial', Grouper(period=image_reader.structure_['dim_z']))])
                                  for metric in METRICS]
    list_of_pipeline_amplitude += [Pipeline([('compute_entropy', PersistenceEntropy()),
                                             ('group_spatial', Grouper(period=image_reader.structure_['dim_z']))])]

    list_of_space_der_pipeline = [Pipeline([('group_diagrams', Grouper(period=image_reader.structure_['dim_z'])),
                                            ('diagram_space_der', MultiDiagramsDerivative(metric=metric,
                                                                                          periodic=True))])
                                  for metric in METRICS]
    list_of_time_der_pipeline = [Pipeline([('shift_resample', ShiftResampler(period=image_reader.structure_['dim_z'])),
                                           ('diagram_time_der', MultiDiagramsDerivative(metric=metric))])
                                 for metric in METRICS]
    if z_fut > 0:
        amplitudes = [pipeline.fit_transform_resample(diagrams[:-z_fut], fake_y[:-z_fut])[0]
                      for pipeline in list_of_pipeline_amplitude]
        space_derivatives = [pipeline.fit_transform_resample(diagrams[:-z_fut], fake_y[:-z_fut])[0]
                             for pipeline in list_of_space_der_pipeline]
    else:
        amplitudes = [pipeline.fit_transform_resample(diagrams, fake_y)[0]
                      for pipeline in list_of_pipeline_amplitude]
        space_derivatives = [pipeline.fit_transform_resample(diagrams, fake_y)[0]
                             for pipeline in list_of_space_der_pipeline]
    time_derivatives = [np.transpose(pipeline.fit_transform_resample(diagrams, fake_y)[0], axes=(1, 0, 2))
                        for pipeline in list_of_time_der_pipeline]
    if result_name_fut is None:
        time_derivatives = [np.concatenate([array, np.zeros((1, *array.shape[1:]))]) for array in time_derivatives]

    features = amplitudes + space_derivatives + time_derivatives
    features = np.concatenate(features, axis=2)
    return features


def process_result_phase_space(result_name_pres, result_name_fut):
    phase_reader = PhaseSpaceReader()
    phase_space_pres = phase_reader.read(SIMULATION_PATH+result_name_pres, tor_angle=np.pi/8, remove_core=False)
    phase_spaces = phase_space_pres
    t_fut = 0
    if result_name_fut is not None:
        phase_space_fut = phase_reader.read(SIMULATION_PATH+result_name_fut, tor_angle=np.pi/8, remove_core=False)
        phase_spaces = np.concatenate([phase_space_pres, np.expand_dims(phase_space_fut[0], axis=0)])
        t_fut = 1
    diagrams_pipeline = Pipeline([('resample', Resampler(period=10)),
                                  ('compute_diagrams', VietorisRipsPersistence())])
    y_fake = np.zeros(phase_spaces.shape[0])
    diagrams = diagrams_pipeline.fit_transform_resample(phase_spaces, y_fake)[0]
    fake_y = np.zeros(len(diagrams))
    amplitude_pipelines = [Pipeline([('scale_diagrams', Scaler(metric=metric)),
                                     ('compute_amplitude', Amplitude(metric=metric))])
                           for metric in METRICS]
    amplitude_pipelines += [Pipeline([('compute_entropy', PersistenceEntropy())])]
    derivative_pipelines = [DiagramDerivative(metric=metric) for metric in METRICS]
    if t_fut > 0:
        amplitudes = [pipeline.fit_transform_resample(diagrams[:-t_fut], fake_y[:-t_fut])[0]
                      for pipeline in amplitude_pipelines]
    else:
        amplitudes = [pipeline.fit_transform_resample(diagrams, fake_y)[0]
                      for pipeline in amplitude_pipelines]

    derivatives = [pipeline.fit_transform(diagrams) for pipeline in derivative_pipelines]
    if result_name_fut is None:
        derivatives = [np.concatenate([array, np.zeros((1, *array.shape[1:]))]) for array in derivatives]
    features = amplitudes + derivatives

    return np.concatenate(features, axis=1)


def process_result_physical_quantities(result_name):
    important_quantities = ['theta', 'temperature', 'temperaturi', 'strmf', 'vpari']
    simulation = Simulation(SIMULATION_PATH+result_name)
    energy = simulation.get_energy()
    important_average_quantities = [simulation.get_average_quantity(q) for q in important_quantities]
    return pd.concat([energy]+important_average_quantities, axis=1)


if __name__ == "__main__":
    dir_path = os.listdir(SIMULATION_PATH)
    dir_path = [path for path in dir_path if ('result' in path)]
    dir_path = sorted(dir_path, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    dir_path.append(None)
    dir_path = dir_path[-3:]
    print(dir_path)
    if not os.path.isfile(SAVE_PATH+'slices_top_features.pickle'):
        images_topology = Parallel(n_jobs=-1)(delayed(process_result_image)(dir_path[t], dir_path[t+1])
                                              for t in range(len(dir_path) - 1))

        images_topology = np.concatenate(images_topology)
        write_pickle(path=SAVE_PATH + 'slices_top_features.pickle', array=images_topology)

    print('phase space')
    if not os.path.isfile(SAVE_PATH+'phase_space_top_features.pickle'):
        phase_space_top = Parallel(n_jobs=-1)(delayed(process_result_phase_space)(dir_path[t], dir_path[t+1])
                                              for t in range(len(dir_path) - 1))

        phase_space_top = np.concatenate(phase_space_top)

        write_pickle(path=SAVE_PATH+'phase_space_top_features.pickle', array=phase_space_top)

    print('extract physics')
    if not os.path.isfile(SAVE_PATH+'physical_features.pickle'):
        physical_qs = Parallel(n_jobs=-1)(delayed(process_result_physical_quantities)(dir_path[t])
                                          for t in range(len(dir_path) - 1))
        physical_qs = pd.concat(physical_qs)
        physical_qs.to_pickle(SAVE_PATH+'physical_features.pickle')


