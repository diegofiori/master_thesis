import giotto as o
from giotto.homology import VietorisRipsPersistence, CubicalPersistence
from giotto.images import DilationFiltration
from giotto.diagram import DiagramDistance
from simulation import Simulation
from exceptions import TimeIDError

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import SpectralClustering as SCModel


class VlasovGiotto:
    """ This class implement the features extraction from simulations using TDA. @TODO add further comments

    Parameters
    ----------
    simulation_path: str
        path to the directory where the simulation is stored. Note that the simulation must be a directory containing
        the results_xxx.h5 files.
    """
    def __init__(self, simulation_path):
        self._simulation_path = simulation_path
        self._files = [name for name in os.listdir(simulation_path)
                       if name.split('.')[-1] == 'h5' and name.split('_')[0] == 'results']
        # think about the possibility to pass a dictionary with the information to pass tp the constructors of
        # topological objects.
        self._filtration = DilationFiltration()
        self._homology = CubicalPersistence()
        self._results = pd.DataFrame()
        self._memory = {}

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

    def _extract_features_from_slices_diagrams(self, diagrams, angles):
        """ The method uses TDA tools in order to create a vector of features.

        The features extracted are
            the inf and sup value for each diagram and available betti number,
            the distance from a predefined 'zero' diagram in different available metrics.
        """
        metrics = ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']
        keys = []
        values = []
        for key in diagrams.keys():
            keys += ['sup_' + str(key) + '_' + str(angle) for angle in angles]
            values.append(diagrams[key][:, :, 1].max(axis=1).reshape(-1, ))
            keys += ['inf_' + str(key) + '_' + str(angle) for angle in angles]
            values.append(diagrams[key][:, :, 1].min(axis=1).reshape(-1, ))
            # add some other key-dependent quantities?

        diagrams = self._add_zero_to_diagrams(diagrams)

        for metric in metrics:
            keys += [metric + '_' + str(angle) for angle in angles]
            values.append(self._compute_diagrams_distance_from_zero(diagrams, metric).reshape(-1, ))

        series = pd.Series(data=np.concatenate(values, axis=0), index=keys)
        return series

    def _compute_diagrams(self, images):
        """ This method should be used internally to the class in order to compute the persistent diagrams for
        2D-images (i.e. matrices)

        Parameters
        ----------
        images: list of 2d numpy.array
            image from which compute the persistence diagrams using the cubical complexes

        Returns
        -------
        diagrams: dict
            a dictionary containing the homology dimensions as keys and the diagram points as values
        """
        images = np.concatenate([np.expand_dims(image, axis=0) for image in images])
        filtered_images = self._filtration.fit_transform(images)
        diagrams = self._homology.fit_transform(filtered_images)

        return diagrams

    @staticmethod
    def _add_zero_to_diagrams(diagrams):
        """ This method add the 'zero' diagram to the dictionary which contains the diagrams information.
        The 'zero' diagram is defined as the one having for all homology dimension only the point (0, 0).

        Parameters
        ----------
        diagrams: dict
        """
        for key in diagrams.keys():
            _, y, z = diagrams[key].shape
            zero_diagram = np.zeros((1, y, z))
            diagrams[key] = np.concatenate([zero_diagram, diagrams[key]])
        return diagrams

    @staticmethod
    def _compute_diagrams_distance_from_zero(diagrams, metric):
        """ The method computes the distance for each diagram from the 'zero' using the specified metric. The 'zero'
        diagram is supposed to be the first diagram.

        Parameters
        ----------
        diagrams: dict
        metric: str
            the metric to be used to compute the distance between diagrams
        """
        distance = DiagramDistance(metric=metric)
        distance_array = distance.fit_transform(diagrams)[1:, 0]
        return distance_array

    def _spectral_clustering(self, simulation):
        time_ids = simulation.time_ids()
        model = SpectralClustering()
        list_of_eigenvalues = [model.get_eigenvalues(simulation.get_phase_space(time_id)) for time_id in time_ids]
        return pd.Series(data=list_of_eigenvalues, index=pd.Index(time_ids))


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










