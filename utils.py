import pickle as pkl
import numpy as np

from giotto.diagrams import Scaler
from giotto.homology import CubicalPersistence
from giotto.pipeline import Pipeline

import matplotlib.pyplot as plt


def read_pickle(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def write_pickle(path, array):
    with open(path, 'wb') as f:
        pkl.dump(array, f)


def plot_slice_diagram(image, homology_dimensions=(0, 1)):
    cub = CubicalPersistence(homology_dimensions=homology_dimensions)
    scaler = Scaler(metric='bottleneck')
    pipeline = Pipeline([('diagram', cub),
                         ('rescale', scaler)])
    diagram = pipeline.fit_transform(np.expand_dims(image, axis=0))

    color_dict = {0: '.r', 1: '.b', 2: '.g'}
    points = diagram[0, :, :-1]
    dims = diagram[0, :, -1]
    plt.figure()
    for hom_dim in homology_dimensions:
        hom_points = points[dims == hom_dim]
        plt.plot(hom_points[:, 0], hom_points[:, 1], color_dict[hom_dim])
    min_b, max_b = np.min(points[:, 0]), np.max(points[:, 0])
    plt.plot([min_b, max_b], [min_b, max_b], 'k')
    plt.show()


def plot_slice_from_h5(h5data, time_id, slice_idx):
    y = h5data['coord1'][()]
    z = h5data['coord2'][()]
    X, Y = np.meshgrid(z, y)
    data = h5data[time_id][()]
    plt.figure()
    slice_data = data[slice_idx]
    plt.scatter(X.reshape(-1, ), Y.reshape(-1, ), c=slice_data.reshape(-1))
    plt.colorbar()
    plt.show()


def plot_slice(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.show()


def moving_average(time_series, av_step):
    new_time_series = np.zeros(time_series.shape)
    for i in range(len(time_series)):
        if i == 0 or av_step == 1:
            new_time_series[i] = time_series[i]
        else:
            new_time_series[i] = np.mean(time_series[max(i-av_step, 0):i], axis=0)
    return new_time_series




