import numpy as np
from simulation import Simulation


class ImageReader(object):
    def __init__(self, physical_q=None):
        self.physical_q = ['theta', 'temperature', 'temperaturi', 'strmf', 'vpari', 'psi'
                           ] if physical_q is None else physical_q
        self.structure_ = None

    def read(self, filename):

        simulation = Simulation(filename)
        time_ids = simulation.time_ids()

        images = [simulation.data[q][time_id] for q in self.physical_q for time_id in time_ids]

        self.structure_ = {'nb_q': len(self.physical_q), 'nb_time_steps': len(time_ids),
                            'dim_z': len(simulation.data[self.physical_q[0]]['coord3']),
                            'inner_sequence': ['dim_z', 'nb_time_steps', 'nb_q']}

        return np.concatenate(images)


class PhaseSpaceReader(object):
    def __init__(self, physical_q=None):
        self.physical_q = ['theta', 'temperature', 'temperaturi', 'strmf', 'vpari', 'psi'
                           ] if physical_q is None else physical_q
        self.structure_ = None

    def read(self, filename, *args, **kwargs):
        simulation = Simulation(filename)
        time_ids = simulation.time_ids()

        phase_spaces = [np.expand_dims(simulation.get_phase_space(time_id, *args, **kwargs), axis=0)
                        for time_id in time_ids]
        phase_spaces = np.concatenate(phase_spaces)

        self.structure_ = {'nb_time_steps': len(time_ids), 'inner_sequence': ['nb_time_steps']}

        return phase_spaces
