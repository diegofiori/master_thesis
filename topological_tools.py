import giotto as o
from giotto.homology import VietorisRipsPersistence, CubicalPersistence
from giotto.images import DilationFiltration
from giotto.diagrams import PairwiseDistance
from simulation import Simulation
from exceptions import TimeIDError

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering as SCModel


class VlasovGiotto:
    """ This class implement the features extraction from simulations using TDA. @TODO add further comments

    Parameters
    ----------
    simulation_path: str
        path to the directory where the simulation is stored. Note that the simulation must be a directory containing
        the results_xxx.h5 files.
    do_spectral_clustering: bool, optional (default=False)
        since the spectral clustering is time consuming it is possible to avoid if setting this flag to False
    """
    def __init__(self, simulation_path, do_spectral_clustering=False):
        self._simulation_path = simulation_path
        self._files = [name for name in os.listdir(simulation_path)
                       if name.split('.')[-1] == 'h5' and name.split('_')[0] == 'results']
        # think about the possibility to pass a dictionary with the information to pass tp the constructors of
        # topological objects.
        self._cub_homology = CubicalPersistence()
        self._vr_homology = VietorisRipsPersistence()
        self._results = pd.DataFrame()
        self._memory = {}
        self._do_sc = do_spectral_clustering

    def show(self):
        """ This method shows the plots for the main physical and topological features presents in the results inner
        dataframe.
        """
        keys_to_show = list(self._results.keys())
        # we need to not show all the eigenvalues of the spectral clustering
        if self._do_sc:
            keys_to_show.remove('spectral_clustering_eig')

        num_plots_y = len(keys_to_show) // 3 + 1
        plt.figure(figsize=(15, 3 * num_plots_y))
        for i, key in enumerate(keys_to_show):
            plt.subplot(num_plots_y, 3, i + 1)
            plt.plot(self._results.reset_index()[key])
            plt.xlabel('time_step')
            plt.ylabel('numerical value')
            plt.title(key)
        plt.show()
        return {i: self._results.index[i] for i in range(len(self._results))}

    def get_topological_features(self, top_q='temperature'):
        """ This is the main method of the class. It extracts some topological features from the simulation. The type
        of the extraction is determined by the top_q parameter (it stays for topological quantity).
        if top_q is a physical quantity the topological extraction is done using cubical complexes built on slices of
        the selected quantities.

        Parameters
        ----------
        top_q: str
            parameter which determine the type of topological extraction.

        Returns
        -------
        a pd.DataFrame containing as indexes the times_id and as columns the topological features
        """
        for name in tqdm(self._files):
            simulation = Simulation(self._simulation_path+name)
            # work on the time_ids
            for time_id in simulation.time_ids():
                if time_id in list(self._memory.keys()):
                    raise TimeIDError(f'The time_id {time_id} is repeated multiple times during the simulation. '
                                      f'If the simulation has been restarted please delete from the folder the '
                                      f'old files.')
                self._memory[time_id] = name
            df = self._extract_topology_from_simulation(simulation, top_q)
            df = pd.concat([df, self._extract_topology_from_phase_space(simulation)], axis=1)
            if self._do_sc:
                df = pd.concat([df, self._spectral_clustering(simulation)], axis=1)

            if self._results.empty:
                self._results = df
            else:
                self._results = pd.concat([self._results, df])

        return self._results

    def add_physical_features(self, phys_qs=None):
        """ The method is studied to be called after the topological features extraction. It adds to the results
        dataframe the columns containing the energy information from the simulations.
        """
        if phys_qs is None:
            phys_qs = ['theta', 'temperature', 'temperaturi', 'strmf', 'vpari', 'psi']
        dfs = []
        for name in tqdm(self._files):
            simulation = Simulation(self._simulation_path+name)
            df = simulation.get_energy()
            for q in phys_qs:
                df[q] = simulation.get_average_quantity(q=q)
            dfs.append(df)

        physical_df = pd.concat(dfs, axis=0)

        if self._results.empty:
            print('Attention: no precomputed topological quantities has been found. '
                  'The code could generate some errors.')
            self._results = physical_df
        else:
            self._results = pd.concat([self._results, physical_df], axis=1)

        return self._results

    def run(self, top_q='temperature'):
        """ This method calls both get_topological_features and add_physical_features in the right order."""
        self.get_topological_features(top_q)
        return self.add_physical_features()

    def _extract_topology_from_simulation(self, simulation, top_q):
        """ This method extracts some topological features from the simulation."""
        df = pd.DataFrame()
        if top_q in list(simulation.data.keys()):
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
            tid_diag = [self._compute_diagrams(timeid_img) for timeid_img in simulation.extract_slices(angles, top_q)]
            df_values = [self._extract_features_from_slices_diagrams(diagrams, angles) for diagrams in tid_diag]
            df = pd.concat(df_values, axis=1).transpose()
            df.index = pd.Index(simulation.time_ids())
        else:
            TypeError(f'The type {top_q} is not supported')
        return df

    def _extract_topology_from_phase_space(self, simulation, reduction=500):
        """ This method extracts topological features from the phase space defined as the space having as axis
        (x, y, psi, all the physical quantities in the simulation.
        """
        time_ids = simulation.time_ids()
        phase_spaces_top = np.concatenate([np.expand_dims(
            self._sample_data(simulation.get_phase_space(time_id, angle=np.pi/4),reduction), axis=0)
                                           for time_id in time_ids], axis=0)
        diagrams = self._compute_phase_spaces_diagrams(phase_spaces_top)
        df = self._extract_features_from_phase_space_diagrams(diagrams, time_ids)
        return df

    def _extract_features_from_phase_space_diagrams(self, diagrams, time_ids):

        metrics = ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']
        series = []
        for key in range(int(diagrams[:, :, 2].max() + 1)):
            mask = diagrams[:, :, 2] == key
            temp_diagrams = diagrams[mask].reshape((diagrams.shape[0], -1, 3))
            temp_series = pd.Series(data=temp_diagrams[:, :, 1].max(axis=1), index=pd.Index(time_ids),
                                    name='vr_sup_'+str(key))
            series.append(temp_series)

        diagrams = self._add_zero_to_diagrams(diagrams)

        for metric in metrics:
            temp_series = pd.Series(data=self._compute_diagrams_distance_from_zero(diagrams, metric),
                                    index=pd.Index(time_ids), name='vr_' + metric)
            series.append(temp_series)

        return pd.concat(series, axis=1)

    def _compute_phase_spaces_diagrams(self, phase_spaces):
        return self._vr_homology.fit_transform(phase_spaces)

    def _extract_features_from_slices_diagrams(self, diagrams, angles):
        """ The method uses TDA tools in order to create a vector of features.

        The features extracted are
            the inf and sup value for each diagram and available betti number,
            the distance from a predefined 'zero' diagram in different available metrics.
        """
        metrics = ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']
        keys = []
        values = []
        for key in range(int(diagrams[:, :, 2].max() + 1)):
            mask = diagrams[:, :, 2] == key
            keys += ['sup_' + str(key) + '_' + str(angle) for angle in angles]
            temp_diagram = diagrams.copy()
            temp_diagram[~mask, 1] = -np.inf
            values.append(temp_diagram[:, :, 1].max(axis=1).reshape(-1, ))
            keys += ['inf_' + str(key) + '_' + str(angle) for angle in angles]
            temp_diagram = diagrams.copy()
            temp_diagram[~mask, 1] = np.inf
            values.append(temp_diagram[:, :, 1].min(axis=1).reshape(-1, ))
            # add some other key-dependent quantities?

        diagrams = self._add_zero_to_diagrams(diagrams)

        for metric in metrics:
            keys += [metric + '_' + str(angle) for angle in angles]
            values.append(self._compute_diagrams_distance_from_zero(diagrams, metric).reshape(-1, ))

        series = pd.Series(data=np.concatenate(values, axis=0), index=keys)
        return series

    def _compute_diagrams(self, images, filtration=None):
        """ This method should be used internally to the class in order to compute the persistent diagrams for
        2D-images (i.e. matrices)

        Parameters
        ----------
        images: list of 2d numpy.array
            image from which compute the persistence diagrams using the cubical complexes

        Returns
        -------
        diagrams: numpy.ndarray
            a numpy array with dimension (N_samples, N_features, 3) where in the last dimension are stored
            (birth, death, homology_dim)
        """
        images = np.concatenate([np.expand_dims(image, axis=0) for image in images])
        if filtration is not None:
            images = filtration.fit_transform(images)
        diagrams = self._cub_homology.fit_transform(images)

        return diagrams

    @staticmethod
    def _add_zero_to_diagrams(diagrams):
        """ This method add the 'zero' diagram to the dictionary which contains the diagrams information.
        The 'zero' diagram is defined as the one having for all homology dimensions only the point (0, 0).

        Parameters
        ----------
        diagrams: numpy.ndarray
        """
        _, y, z = diagrams.shape
        zero_diagram = np.zeros((1, y, z))
        # now we need to take the same structure used for the other diagrams (same number of features per homology dim)
        max_homology_dim = diagrams[:, :, 2].max()
        nb_features = [np.sum(diagrams[0, :, 2] == float(hom_dim)) for hom_dim in range(int(max_homology_dim) + 1)]
        for i in range(1, len(nb_features)):
            zero_diagram[0, nb_features[i-1]:nb_features[i-1]+nb_features[i], 2] = i
        diagrams = np.concatenate([zero_diagram, diagrams])
        return diagrams

    @staticmethod
    def _compute_diagrams_distance_from_zero(diagrams, metric):
        """ The method computes the distance for each diagram from the 'zero' using the specified metric. The 'zero'
        diagram is supposed to be the first diagram.

        Parameters
        ----------
        diagrams: numpy.ndarray
        metric: str
            the metric to be used to compute the distance between diagrams
        """
        distance = PairwiseDistance(metric=metric)
        distance_array = distance.fit_transform(diagrams)[1:, 0]
        return distance_array

    def _spectral_clustering(self, simulation, reduction=1000):
        """ This method uses the spectal clustering algorithm and returns the eigenvalues of the Laplacian of the
        graph built on the simulation phase space points cloud.
        """
        time_ids = simulation.time_ids()
        model = SpectralClustering()
        list_of_eigenvalues = [model.get_eigenvalues(self._sample_data(simulation.get_phase_space(time_id), reduction))
                               for time_id in time_ids]
        return pd.Series(data=list_of_eigenvalues, index=pd.Index(time_ids), name='spectral_clustering_eig')

    @staticmethod
    def _sample_data(data, reduction=100, random=False):
        """ the method samples from the data a subsample with dimension len(data) // reduction."""

        if random:
            indx = np.random.choice(len(data), len(data)//reduction)
        else:
            indx = np.arange(0, len(data), reduction)
        return data[indx]

    @classmethod
    def from_df(cls, df, path='', do_spectral_clustering=False, memory=None):
        """ The idea of this method is to load a dataframe containing as columns with the
        topological (and/or physical) features precomputed, as indexes the time ids.

        Parameters
        ----------
        df: pd.DataFrame
            the precomputed dataframe containing relevant features.
        path: str, optional (default = '')
            the path where are loaded the simulations from which the features are extracted. If not specified it
            would not be possible to extract further features, only use the show() method.
        do_spectral_clustering: bool, optional (default=False)
            value passed to the constructor of the class
        memory: dict, optional (default=None)
            if the memory dictionary is not passed a new dict will be created with keys the df.index and values the
            string 'not_available'
        """
        if memory is None:
            memory = {key: 'not available' for key in df.index}

        assert set(df.index) == set(memory.keys())

        self = cls(path, do_spectral_clustering)
        self._memory = memory
        self._results = df
        return self


class SpectralClustering:
    """ The class implement the spectral clustering method. Note that the Scikit-learn class has not been used since
    it does not provide the graph Laplacian eigenvalues and eigenvectors.
    """
    def __init__(self, n_neighbors=5, affinity='rbf'):
        self._n_neighbors = n_neighbors
        self._affinity = affinity
        self._eigenvalues = None
        self._adjency = None

    def fit(self, X):
        # compute the graph
        spectral_clustering_model = SCModel(affinity=self._affinity, n_neighbors=self._n_neighbors)
        spectral_clustering_model.fit(X)
        adjency = spectral_clustering_model.affinity_matrix_
        self._adjency = adjency

        # compute the Graph Laplacian
        degree_matrix = np.diag(adjency.sum(axis=1))
        laplacian = degree_matrix - adjency

        # Now we can compute the eigenvalues and the eigenvectors
        eig_val, _ = np.linalg.eig(laplacian)
        self._eigenvalues = eig_val

        return self

    def predict(self, X, n_clusters=8):
        spectral_clustering_model = SCModel(n_clusters=n_clusters, affinity='precomputed')
        labels = spectral_clustering_model.fit_predict(self._adjency)
        return labels

    @property
    def eigenvalues(self):
        return np.sort(self._eigenvalues)

    def get_eigenvalues(self, X):
        self.fit(X)
        return self.eigenvalues










