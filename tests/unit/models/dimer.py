import unittest
from tests.definitions import ROOT_DIR
from tests.infrastructure.load import load_sparse_matrix
from oqspy.models.dimer import \
    dimer_get_sys_size,\
    dimer_get_hamiltonian,\
    dimer_get_periods, \
    dimer_get_driving_hamiltonias, \
    dimer_get_driving_functions
from scipy.sparse.linalg import norm as sps_mtx_norm
import numpy as np


class DimerModel:
    def __init__(self, case):

        self.num_particles = 10
        self.sys_size = 11

        self.diss_type = 1
        self.diss_gamma = 0.1

        if case == 1:
            self.E = 0.0
            self.U = 0.5
            self.J = 1.0

            self.drv_type = 1
            self.drv_ampl = 3.4
            self.drv_freq = 1.0
            self.drv_phas = 0.0
        else:
            self.E = 1.0
            self.U = 0.5
            self.J = 1.0

            self.drv_type = 0
            self.drv_ampl = 1.5
            self.drv_freq = 1.0
            self.drv_phas = 0.0

    def get_path(self):
        path = f'{ROOT_DIR}/fixtures/models/dimer/'
        return path

    def get_suffix(self):
        precision = f'.{4}f'
        suffix = f'_np({self.num_particles})' + \
                 f'_diss({self.diss_type}_{self.diss_gamma:{precision}})' + \
                 f'_prm({self.E:{precision}}_{self.U:{precision}}_{self.J:{precision}})' + \
                 f'_drv({self.diss_type}_{self.drv_ampl:{precision}}_{self.drv_freq:{precision}}_{self.drv_phas:{precision}})' + \
                 '.txt'
        return suffix


class TestDimerModel(unittest.TestCase):

    def setUp(self):
        self.dimer_1 = DimerModel(1)
        self.dimer_2 = DimerModel(2)

    def tearDown(self):
        pass

    def test_correct_size_of_hamiltonian(self):
        sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
        self.assertEqual(self.dimer_1.sys_size, sys_size)

    def test_period_correctness(self):
        period_expected = [2.0 * np.pi / self.dimer_1.drv_freq]
        period_actual = dimer_get_periods(self.dimer_1.drv_freq)
        self.assertListEqual(period_expected, period_actual)

    def test_hamiltonian_correctness(self):
        fn = self.dimer_1.get_path() + 'hamiltonian_mtx' + self.dimer_1.get_suffix()
        h_expected = load_sparse_matrix(fn, self.dimer_1.sys_size)
        h_actual = dimer_get_hamiltonian(
            self.dimer_1.num_particles,
            self.dimer_1.E,
            self.dimer_1.U,
            self.dimer_1.J
        )
        norm_diff = sps_mtx_norm(h_expected - h_actual)
        self.assertLess(norm_diff, 1.0e-14)

    def test_driving_hamiltonians_correctness(self):
        fn = self.dimer_1.get_path() + 'hamiltonian_drv_mtx' + self.dimer_1.get_suffix()
        h_expected = load_sparse_matrix(fn, self.dimer_1.sys_size)
        hs_actual = dimer_get_driving_hamiltonias(
            self.dimer_1.num_particles,
        )
        norm_diff = sps_mtx_norm(h_expected - hs_actual[0])
        self.assertLess(norm_diff, 1.0e-14)

    def test_driving_functions(self):
        period = dimer_get_periods(self.dimer_1.drv_freq)[0]
        times = list(np.linspace(start=0.0, stop=period, num=11))
        drv_expected = []
        drv_actual = []
        driving_function = dimer_get_driving_functions(
            self.dimer_1.drv_type,
            self.dimer_1.drv_ampl,
            self.dimer_1.drv_freq,
            self.dimer_1.drv_phas
        )[0]
        for time in times:
            drv_expected.append(self.dimer_1.drv_ampl * np.sin(self.dimer_1.drv_freq * time + self.dimer_1.drv_phas))
            drv_actual.append(driving_function(time))
        self.assertListEqual(drv_expected, drv_actual)
