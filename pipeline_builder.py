import numpy as np
from giotto.diagrams import PersistenceEntropy, Amplitude
from giotto.homology import VietorisRipsPersistence, CubicalPersistence
from giotto.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion, make_union

from filter import FilterBigComponents
from resampler import Grouper, Degrouper
from diagram_derivatives import MultiDiagramsDerivative, DiagramDerivative
from giotto.diagrams import BettiCurve, HeatKernel, PersistenceLandscape
from masker import Masker, Squeezer
from resampler import Resampler
from utils import SemInterval
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

HEAT_LIST = [
    {'sigma': 1.6, 'n_values': 50},
    {'sigma': 3.2, 'n_values': 50}
]


def build_the_space_pipeline(space_period, remove_n_comp=0, n_jobs=None):
    cubical = CubicalPersistence(homology_dimensions=[0, 1])
    scaler = Scaler(metric='bottleneck')
    filter_bc = FilterBigComponents(n_filter=remove_n_comp)
    initial_pipeline_elements = [cubical,  scaler, filter_bc]
    entropy_step = [initial_pipeline_elements + [PersistenceEntropy(), 'persistence_entropy']]
    betti_step = [initial_pipeline_elements + [BettiCurve(n_values=50), Degrouper(dim=1), 'betti_curve']]
    landscape_step = [initial_pipeline_elements + [PersistenceLandscape(n_layers=10, n_values=50), Degrouper(dim=1),
                                                   Degrouper(dim=1), 'pers_landscape']]
    heat_steps = [initial_pipeline_elements + [HeatKernel(**heat_dict), Degrouper(dim=1),
                                               Degrouper(dim=1), f'heat_kernel_{heat_dict["sigma"]}']
                  for heat_dict in HEAT_LIST]
    amplitude_steps = [initial_pipeline_elements+[Amplitude(**metric, order=None),
        f'amplitude_{metric["metric"]}_{"_".join(list(map(str, metric["metric_params"].values())))}']
                       for metric in METRIC_LIST]
    amplitudes_entropy_steps = entropy_step + amplitude_steps
    # amplitudes_pipeline = make_union(*[make_pipeline(*steps) for steps in amplitudes_entropy_steps], n_jobs=n_jobs)
    # now we build the derivative pipeline
    grouper = Grouper(period=space_period)
    degrouper = Degrouper()
    derivative_steps = [initial_pipeline_elements +
                        [grouper, MultiDiagramsDerivative(**metric, periodic=True, order=None),
        degrouper, f'derivative_{metric["metric"]}_{"_".join(list(map(str, metric["metric_params"].values())))}']
                        for metric in DERIVATIVE_METRIC_LIST]
    all_steps = amplitudes_entropy_steps + derivative_steps + heat_steps + betti_step + landscape_step
    total_pipeline = FeatureUnion([(steps[-1], make_pipeline(*steps[:-1])) for steps in all_steps], n_jobs=n_jobs)

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


def get_pipeline_index():
    dict_transf = {}
    index = 0
    pipeline = build_the_space_pipeline(80)
    list_transf = [x[0] for x in pipeline.transformer_list]
    for transf in list_transf:
        if transf.split('_')[0] in ['amplitude', 'derivative', 'persistence']:
            dict_transf[transf] = SemInterval(index, index + 2)
            index += 2
        elif transf.split('_')[0] == 'betti':
            dict_transf[transf] = SemInterval(index, index + 100)
            index += 100
        elif transf.split('_')[0] == 'heat':
            dict_transf[transf] = SemInterval(index, index + 2 * 50 * 50)
            index += 50 * 50 * 2
        elif transf.split('_')[0] == 'pers':
            dict_transf[transf] = SemInterval(index, index + 2 * 10 * 50)
            index += 2 * 10 * 50
        else:
            raise KeyError(f'The key {transf} is not supported')

    return dict_transf
