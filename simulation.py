import numpy as np
import pandas as pd
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

    @staticmethod
    def _compute_perp_grad(f, xs, ys):
        """ The method computes numerically the perpendicular gradient of function f. f should be a 3D tensor with
        dimensions (phi, x, y). The method returns a tuple with elements df/dx and df/dy.
        Parameters
        ----------
        f: np.array
            the function whose gradient is needed
        xs: np.array
            the x-coordinates value, corresponding to the second dimension of f, i.e. f[:, i, :] has x-coordinate xs[i]
        ys: np.array
            the y-coordinates value, corresponding to the second dimension of f, i.e. f[:, :, i] has y-coordinate ys[i]
        """
        xs = np.concatenate([xs.reshape(-1, ), np.array([0]).reshape(-1, )])
        dx = xs - np.roll(xs, 1)
        dx = dx[1:].reshape((1, -1, 1))
        ys = np.concatenate([ys.reshape(-1, ), np.array([0]).reshape(-1, )])
        dy = ys - np.roll(ys, 1)
        dy = dy[1:].reshape((1, 1, -1))
        n_phi, n_x, n_y = f.shape
        new_f = np.concatenate([f, np.zeros((n_phi, 1, n_y))], axis=1)
        dfx = new_f - np.roll(new_f, 1, axis=1)
        new_f = np.concatenate([f, np.zeros((n_phi, n_x, 1))], axis=2)
        dfy = new_f - np.roll(new_f, 1, axis=2)
        return dfx[:, 1:, :]/dx, dfy[:, :, 1:]/dy

    @staticmethod
    def _compute_volume_integral(f, xs, ys, psis, R):
        """ The method computes numerically the volume integral of the function f. f should be a 3D tensor with
        dimensions (psi, x, y). The method returns the value of the integral.

        Parameters
        ----------
        f: np.array
            the function whose integral is needed
        xs: np.array
            the x-coordinates value, corresponding to the second dimension of f, i.e. f[:, i, :] has x-coordinate xs[i]
        ys: np.array
            the y-coordinates value, corresponding to the second dimension of f, i.e. f[:, :, i] has y-coordinate ys[i]
        phis: np.array
            the phi-coordinates value, corresponding to the second dimension of f,
            i.e. f[i, :, :] has phi-coordinate phis[i]
        """
        xs = np.concatenate([xs.reshape(-1, ), np.array([xs[-1]]).reshape(-1, )])
        # in this way we exclude the boundaries of the domain
        dx = xs - np.roll(xs, 1)
        dx = dx[1:].reshape((1, -1, 1))
        ys = np.concatenate([ys.reshape(-1, ), np.array([ys[-1]]).reshape(-1, )])
        dy = ys - np.roll(ys, 1)
        dy = dy[1:].reshape((1, 1, -1))
        psis = np.concatenate([psis.reshape(-1, ), np.array([psis[-1]]).reshape(-1, )])
        dpsi = psis - np.roll(psis, 1)
        dpsi = dpsi[1:].reshape((-1, 1, 1))
        diff_correction = (xs + R)[:-1].reshape((1, -1, 1))
        dx = dx * diff_correction
        differential_prod = dx * dy * dpsi

        return np.sum(f*differential_prod)

    def _compute_energy_contributions(self, time_id):
        """ The method computes the total energy and each single contribution for a given time_id.
        The energy is supposed to be the sum of 4 different sources: the kinetic term (given by the parallel
        ion velocity), the pressure term, the electrostatic term and the magnetic term.
        Note that for a GBS simulation the total energy should be constant as proved in the paper by Zeiler in 1997.

        Parameters
        ----------
        time_id: str
            the time_id in which the energy is needed

        Returns
        -------
        a pd.Series containing both the total energy and its contribution terms, with the following index ordering
            ['v_pari_energy', 'pressure_energy', 'electrostatic_energy', 'magnetic_energy', 'total_energy']
        """
        phi = self.data['strmf'][time_id]
        psi = self.data['psi'][time_id]
        T_e = self.data['temperature'][time_id]
        T_i = self.data['temperaturi'][time_id]
        n = np.exp(self.data['theta'][time_id])
        v_pari = self.data['vpari'][time_id]
        # compute energy contributions
        # v_pari
        integrand = n * v_pari**2
        v_pari_energy = self._compute_volume_integral(integrand, self.data['theta']['coord1'],
                                                      self.data['theta']['coord2'],
                                                      self.data['theta']['coord3'], R=1)
        # pressure contribution
        tau = 1
        integrand = 3 * n * (T_e + tau * T_i)
        pressure_energy = self._compute_volume_integral(integrand, self.data['temperaturi']['coord1'],
                                                        self.data['temperaturi']['coord2'],
                                                        self.data['temperaturi']['coord3'], R=1)

        # Electrostatic contribution
        n_0 = 1
        temp_function = phi + n_0 * tau * n * T_i

        df_dx, df_dy = self._compute_perp_grad(temp_function,
                                               self.data['strmf']['coord1'],
                                               self.data['strmf']['coord2'])
        integrand = df_dx**2 + df_dy**2
        electrostatic_energy = self._compute_volume_integral(integrand, self.data['strmf']['coord1'],
                                                             self.data['strmf']['coord2'], self.data['strmf']['coord3'],
                                                             R=1)

        # magnetic contribution
        dpsi_dx, dpsi_dy = self._compute_perp_grad(psi, self.data['psi']['coord1'], self.data['psi']['coord2'])
        integrand = dpsi_dx**2 + dpsi_dy**2
        magnetic_energy = self._compute_volume_integral(integrand, self.data['psi']['coord1'],
                                                        self.data['psi']['coord2'], self.data['psi']['coord3'], R=1)

        total_energy = v_pari_energy + pressure_energy + electrostatic_energy + magnetic_energy

        return pd.Series(data=[v_pari_energy, pressure_energy, electrostatic_energy, magnetic_energy, total_energy],
                         index=pd.Index(['v_pari_energy', 'pressure_energy', 'electrostatic_energy',
                                         'magnetic_energy', 'total_energy']), name=time_id)

    def get_energy(self):
        """ This method produce a pandas DataFrame containing all the enrgy contribution per each time_id store inside
        the simulation file.
        """
        df = pd.concat([self._compute_energy_contributions(time_id) for time_id in self.time_ids()], axis=1).transpose()
        return df

    def get_average_quantity(self, q='theta', point_coordinate=None):
        """ This method returns a pandas Series containing the average (on psi, *axis) values for each time snapshot.

        Parameters
        ----------
        q: str
            the quantity that has to be monitored
        point_coordinate: tuple, optional (default None)
            if given the average is taken only on the psi axis, and the coordinate passed is monitored.
        """
        time_ids = self.time_ids()

        def compute_mean(array):
            if point_coordinate is None:
                return float(array.mean())
            else:
                return array.mean(axis=0)[point_coordinate[0], point_coordinate[1]]

        return pd.Series(data=[compute_mean(self.data[q][time_id]) for time_id in time_ids], index=pd.Index(time_ids))

    def get_phase_space(self, time_id, coords=None):
        """ The method gives the discretized phase space of the fluid (plasma) in the simulation. It is possible both
        impose the coordinates axis to use or use all the information available. Since we only have the information
        at the grid points we call phase space the space having the physical quantities as axis. Each point corresponds
        to a grid point (x, y, psi) [or (psi, x, y) speaking with the formalism of the data]"""
        if coords is None:
            coords = list(set(self.data.keys()) - {'time', 'cstep'})
        n_z, n_x, n_y = self.data[coords[0]][time_id].shape

        z = self.data[coords[0]]['coord3'].reshape((-1, 1, 1))
        x = self.data[coords[0]]['coord1'].reshape((1, -1, 1))
        y = self.data[coords[0]]['coord2'].reshape((1, 1, -1))

        def flatten_grid(array, axis_rep):

            array = np.repeat(array, axis_rep[0], axis=0)
            array = np.repeat(array, axis_rep[1], axis=1)
            array = np.repeat(array, axis_rep[2], axis=2)

            return array.flatten()

        z_flat = np.expand_dims(flatten_grid(z, (1, n_x, n_y)), axis=1)
        y_flat = np.expand_dims(flatten_grid(y, (n_z, n_x, 1)), axis=1)
        x_flat = np.expand_dims(flatten_grid(x, (n_z, 1, n_y)), axis=1)
        list_of_vertical_features = [z_flat, x_flat, y_flat]
        for coord in coords:
            list_of_vertical_features.append(np.expand_dims(self.data[coord][time_id].flatten(), axis=1))

        features = np.concatenate(list_of_vertical_features, axis=1)

        return features



