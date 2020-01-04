import numpy as np
from giotto.diagrams import PersistenceEntropy, Amplitude
from giotto.homology import VietorisRipsPersistence, CubicalPersistence
from giotto.pipeline import make_pipeline
from sklearn.pipeline import make_union

from resampler import Grouper, Degrouper
from diagram_derivatives import MultiDiagramsDerivative, DiagramDerivative
from masker import Masker, Squeezer
from resampler import Resampler
from diagram_scaler import Scaler


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


def build_the_space_pipeline(space_period, n_jobs=None):
    cubical = CubicalPersistence(homology_dimensions=[0, 1])
    scaler = Scaler(metric='bottleneck')
    initial_pipeline_elements = [cubical,  scaler]
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


def build_phase_space_pipeline(resemple_period=1, n_jobs=None):
    resampler = Resampler(period=resemple_period)
    rips = VietorisRipsPersistence(homology_dimensions=[0, 1])
    scaler = Scaler(metric='bottleneck')
    initial_pipeline_elements = [resampler, rips, scaler]
    entropy_step = [initial_pipeline_elements + [Masker(), PersistenceEntropy()]]
    amplitude_steps = [initial_pipeline_elements + [Masker(), Amplitude(**metric, order=None)]
                       for metric in METRIC_LIST]
    amplitudes_entropy_steps = entropy_step + amplitude_steps
    time_derivative_steps = [initial_pipeline_elements + [DiagramDerivative(**metric, periodic=False, order=None),
                                                          Squeezer()]
                             for metric in DERIVATIVE_METRIC_LIST]
    all_steps = amplitudes_entropy_steps + time_derivative_steps
    total_pipeline = make_union(*[make_pipeline(*steps) for steps in all_steps], n_jobs=n_jobs)
    return total_pipeline


