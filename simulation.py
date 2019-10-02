import numpy as np
import h5py
import math
import matplotlib.pyplot as plt


class Simulation:
    """ This class deals with the physics underlying the project. All the physical quantities needed and derived from
    the simulation are computed inside this class. The class reads an h5 file and store all the information in a
    pythonic format.
    """
    def __init__(self, path, dim=3):
        with h5py.File(path, 'r') as f:
            reference_num = int(''.join(
                list(filter(str.isdigit, path.split('/')[-1].split('.')[-2]))))  # extract the number from the name
            self._setup_file = f['files'][f'STDIN.{reference_num}'][()][0].decode('ascii')
            self._data = self._extract_data(f['data'], dim=dim)
            self._input = self._recursive_data_mod(f['data'][f'input.{reference_num}'])

    @property
    def data(self):
        return self._data

    @property
    def setup_file(self):
        return self._setup_file

    def _extract_data(self, data, dim=3):
        data = data[f'var{dim}d']
        return self._recursive_data_mod(data)

    def _recursive_data_mod(self, data):
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, h5py._hl.group.Group):
            data = dict(data)
            for key, value in data.items():
                data[key] = self._recursive_data_mod(value)
        elif isinstance(data, h5py._hl.dataset.Dataset):
            data = data[()]
        return data

    @staticmethod
    def _plot_slices(data_dict, slices_idx, time_id):
        x = data_dict['coord3']
        y = data_dict['coord1']
        z = data_dict['coord2']
        X, Y = np.meshgrid(z, y)
        data = data_dict[time_id]
        n_y = math.ceil(len(slices_idx) / 3)
        plt.figure(figsize=(3 * 5, n_y * 3))
        for i, slice_idx in enumerate(slices_idx):
            slice_data = data[slice_idx]
            plt.subplot(3, n_y, i + 1)
            plt.scatter(X.reshape(-1, ), Y.reshape(-1, ), c=slice_data.reshape(-1))
        plt.colorbar()
        plt.show()

    def show(self, q='temperature', spatial_step=10):
        time_ids = [id_ for id_ in self.data[q].keys() if id_[0].isdigit()]
        data = self.data[q]
        slices_idx = [i for i in range(0, len(data['coord3']), spatial_step)]
        for time_id in time_ids:
            self._plot_slices(data, slices_idx, time_id)

    def extract_slices(self, toroidal_angles, q='temperature'):
        # check if the angles are in the right format
        toroidal_angles = [angle - math.floor(angle / 2 / np.pi) * 2 * np.pi if angle > 0
                           else angle - math.ceil(angle / 2 / np.pi) * 2 * np.pi for angle in toroidal_angles]
        time_ids = [key for key in self.data[q].keys() if key[0].isdigit()]

        def get_slice(angle, time_id):
            sim_tor_angles = self.data[q]['coord3']
            idx = np.abs(sim_tor_angles - angle).argmin()
            return self.data[q][time_id][idx]

        slices = [[get_slice(angle, time_id) for angle in toroidal_angles] for time_id in time_ids]
        return slices

    def time_ids(self):
        return [key for key in self.data['temperature'].keys() if key[0].isdigit()]
