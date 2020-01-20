import unittest
from oqspy.oqs import oqs
from scipy.sparse import csr_matrix
import numpy as np
from tests.unit.models.dimer import DimerModel
from tests.infrastructure.load import load_sparse_matrix
from scipy.sparse.linalg import norm as sps_mtx_norm
from oqspy.models.dimer import \
    dimer_get_sys_size,\
    dimer_get_hamiltonian,\
    dimer_get_driving_hamiltonias, \
    dimer_get_driving_functions, \
    dimer_get_dissipators


class TestOQS(unittest.TestCase):

    def setUp(self):
        self.dimer_1 = DimerModel(1)
        self.dimer_2 = DimerModel(2)

    def tearDown(self):
        pass

    def test_init(self):
        with self.assertRaises(TypeError):
            oqs('aaa', 10, 10)
        with self.assertRaises(TypeError):
            oqs(10, 'aaa', 10)
        with self.assertRaises(TypeError):
            oqs(10, 10, 'aaa')

        with self.assertRaises(ValueError):
            oqs(0, 10, 10)
        with self.assertRaises(ValueError):
            oqs(10, -1, 10)
        with self.assertRaises(ValueError):
            oqs(10, 10, -1)

    def test_init_hamiltonian(self):
        sys = oqs(10, 1, 1)
        with self.assertRaises(TypeError):
            sys.init_hamiltonian('aaa')

        rows = np.array([0, 0, 1, 2, 2, 2])
        cols = np.array([0, 2, 2, 0, 1, 2])
        vals = np.array([1, 2, 3, 4, 5, 6])
        h = csr_matrix((vals, (rows, cols)), shape=(3, 3))
        with self.assertRaises(ValueError):
            sys.init_hamiltonian(h)

        # Dimer: 1
        fn = self.dimer_1.get_path() + 'hamiltonian_mtx' + self.dimer_1.get_suffix()
        h_expected = load_sparse_matrix(fn, self.dimer_1.sys_size)
        sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
        hamiltonian = dimer_get_hamiltonian(
            self.dimer_1.num_particles,
            self.dimer_1.E,
            self.dimer_1.U,
            self.dimer_1.J
        )
        sys = oqs(sys_size, 0, 1)
        sys.init_hamiltonian(hamiltonian)
        h_actual = sys._oqs__hamiltonian
        norm_diff = sps_mtx_norm(h_expected - h_actual)
        self.assertLess(norm_diff, 1.0e-14)

        # Dimer: 2
        fn = self.dimer_2.get_path() + 'hamiltonian_mtx' + self.dimer_2.get_suffix()
        h_expected = load_sparse_matrix(fn, self.dimer_2.sys_size)
        sys_size = dimer_get_sys_size(self.dimer_2.num_particles)
        hamiltonian = dimer_get_hamiltonian(
            self.dimer_2.num_particles,
            self.dimer_2.E,
            self.dimer_2.U,
            self.dimer_2.J
        )
        sys = oqs(sys_size, 0, 1)
        sys.init_hamiltonian(hamiltonian)
        h_actual = sys._oqs__hamiltonian
        norm_diff = sps_mtx_norm(h_expected - h_actual)
        self.assertLess(norm_diff, 1.0e-14)

    def test_init_driving_hamiltonians(self):
        sys = oqs(10, 1, 1)

        def driving_1p(time):
            drv = 1.5 * np.sin(1.0 * time + 0.0)
            return drv

        def driving_2p(p1, p2):
            drv = p1 + p2
            return drv

        rows = np.array([0, 0, 1, 2, 2, 2])
        cols = np.array([0, 2, 2, 0, 1, 2])
        vals = np.array([1, 2, 3, 4, 5, 6])
        h_1 = csr_matrix((vals, (rows, cols)), shape=(3, 3))
        h_2 = csr_matrix((vals, (rows, cols)), shape=(10, 10))

        with self.assertRaises(TypeError):
            sys.init_driving('aaa', [driving_1p])
        with self.assertRaises(TypeError):
            sys.init_driving(['aaa'], [driving_1p])
        with self.assertRaises(TypeError):
            sys.init_driving(h_1, [driving_1p])
        with self.assertRaises(ValueError):
            sys.init_driving([h_1], [driving_1p])
        with self.assertRaises(ValueError):
            sys.init_driving([h_1, h_1], [driving_1p])
        with self.assertRaises(ValueError):
            sys.init_driving([h_2, h_2], [driving_1p])

        with self.assertRaises(TypeError):
            sys.init_driving([h_2], 'aaa')
        with self.assertRaises(TypeError):
            sys.init_driving([h_2], ['aaa'])
        with self.assertRaises(TypeError):
            sys.init_driving([h_2], driving_1p)
        with self.assertRaises(ValueError):
            sys.init_driving([h_2], [driving_2p])
        with self.assertRaises(ValueError):
            sys.init_driving([h_2], [driving_1p, driving_1p])
        with self.assertRaises(ValueError):
            sys.init_driving([h_2], [driving_2p, driving_2p])

        # Dimer: 1
        fn = self.dimer_1.get_path() + 'hamiltonian_drv_mtx' + self.dimer_1.get_suffix()
        h_expected = load_sparse_matrix(fn, self.dimer_1.sys_size)
        sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
        hamiltonians = dimer_get_driving_hamiltonias(self.dimer_1.num_particles)
        functions = dimer_get_driving_functions(
            self.dimer_1.drv_type,
            self.dimer_1.drv_ampl,
            self.dimer_1.drv_freq,
            self.dimer_1.drv_phas
        )
        sys = oqs(sys_size, 1, 1)
        sys.init_driving(hamiltonians, functions)
        h_actual = sys._oqs__driving_hamiltonians[0]
        norm_diff = sps_mtx_norm(h_expected - h_actual)
        self.assertLess(norm_diff, 1.0e-14)

        # Dimer: 2
        fn = self.dimer_2.get_path() + 'hamiltonian_drv_mtx' + self.dimer_2.get_suffix()
        h_expected = load_sparse_matrix(fn, self.dimer_2.sys_size)
        sys_size = dimer_get_sys_size(self.dimer_2.num_particles)
        hamiltonians = dimer_get_driving_hamiltonias(self.dimer_2.num_particles)
        functions = dimer_get_driving_functions(
            self.dimer_2.drv_type,
            self.dimer_2.drv_ampl,
            self.dimer_2.drv_freq,
            self.dimer_2.drv_phas
        )
        sys = oqs(sys_size, 1, 1)
        sys.init_driving(hamiltonians, functions)
        h_actual = sys._oqs__driving_hamiltonians[0]
        norm_diff = sps_mtx_norm(h_expected - h_actual)
        self.assertLess(norm_diff, 1.0e-14)

    def test_init_dissipation(self):
        sys = oqs(10, 1, 1)

        rows = np.array([0, 0, 1, 2, 2, 2])
        cols = np.array([0, 2, 2, 0, 1, 2])
        vals = np.array([1, 2, 3, 4, 5, 6])
        d_1 = csr_matrix((vals, (rows, cols)), shape=(3, 3))
        d_2 = csr_matrix((vals, (rows, cols)), shape=(10, 10))

        with self.assertRaises(TypeError):
            sys.init_dissipation('aaa', [0.1])
        with self.assertRaises(TypeError):
            sys.init_dissipation(['aaa'], [0.1])
        with self.assertRaises(TypeError):
            sys.init_dissipation(d_1, [0.1])
        with self.assertRaises(ValueError):
            sys.init_dissipation([d_1], [0.1])
        with self.assertRaises(ValueError):
            sys.init_dissipation([d_1, d_1], [0.1])
        with self.assertRaises(ValueError):
            sys.init_dissipation([d_2, d_2], [0.1])

        with self.assertRaises(TypeError):
            sys.init_dissipation([d_2], 'aaa')
        with self.assertRaises(TypeError):
            sys.init_dissipation([d_2], ['aaa'])
        with self.assertRaises(TypeError):
            sys.init_dissipation([d_2], 0.1)
        with self.assertRaises(ValueError):
            sys.init_dissipation([d_2], [-0.1])
        with self.assertRaises(ValueError):
            sys.init_dissipation([d_2], [-0.1, -0.1])
        with self.assertRaises(ValueError):
            sys.init_dissipation([d_2], [0.1, 0.1])

        # Dimer: 1
        fn = self.dimer_1.get_path() + 'diss_0_mtx' + self.dimer_1.get_suffix()
        diss_expected = load_sparse_matrix(fn, self.dimer_1.sys_size)
        sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
        dissipators = dimer_get_dissipators(self.dimer_1.num_particles)
        sys = oqs(sys_size, 1, 1)
        sys.init_dissipation(dissipators, [0.1])
        diss_actual = sys._oqs__dissipators[0]
        norm_diff = sps_mtx_norm(diss_expected - diss_actual)
        self.assertLess(norm_diff, 1.0e-14)

        # Dimer: 2
        fn = self.dimer_2.get_path() + 'diss_0_mtx' + self.dimer_2.get_suffix()
        diss_expected = load_sparse_matrix(fn, self.dimer_2.sys_size)
        sys_size = dimer_get_sys_size(self.dimer_2.num_particles)
        dissipators = dimer_get_dissipators(self.dimer_2.num_particles)
        sys = oqs(sys_size, 1, 1)
        sys.init_dissipation(dissipators, [0.1])
        diss_actual = sys._oqs__dissipators[0]
        norm_diff = sps_mtx_norm(diss_expected - diss_actual)
        self.assertLess(norm_diff, 1.0e-14)

    def test_calc_lindbladian(self):
        with self.assertRaises(ValueError):
            sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
            sys = oqs(sys_size, 0, 1)
            sys._oqs__calc_lindbladian()

        # Dimer: 1
        sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
        hamiltonian = dimer_get_hamiltonian(
            self.dimer_1.num_particles,
            self.dimer_1.E,
            self.dimer_1.U,
            self.dimer_1.J
        )
        dissipators = dimer_get_dissipators(self.dimer_1.num_particles)
        fn = self.dimer_1.get_path() + 'lindbladian_mtx' + self.dimer_1.get_suffix()
        l_expected = load_sparse_matrix(fn, self.dimer_1.sys_size * self.dimer_1.sys_size)
        sys = oqs(sys_size, 0, 1)
        sys.init_hamiltonian(hamiltonian)
        sys.init_dissipation(dissipators, [0.1 / float(self.dimer_1.num_particles)])
        sys._oqs__calc_lindbladian()
        l_actual = sys._oqs__lindbladian
        norm_diff = sps_mtx_norm(l_expected - l_actual)
        self.assertLess(norm_diff, 1.0e-14)

        # Dimer: 2
        sys_size = dimer_get_sys_size(self.dimer_2.num_particles)
        hamiltonian = dimer_get_hamiltonian(
            self.dimer_2.num_particles,
            self.dimer_2.E,
            self.dimer_2.U,
            self.dimer_2.J
        )
        dissipators = dimer_get_dissipators(self.dimer_2.num_particles)
        fn = self.dimer_2.get_path() + 'lindbladian_mtx' + self.dimer_2.get_suffix()
        l_expected = load_sparse_matrix(fn, self.dimer_2.sys_size * self.dimer_2.sys_size)
        sys = oqs(sys_size, 0, 1)
        sys.init_hamiltonian(hamiltonian)
        sys.init_dissipation(dissipators, [0.1 / float(self.dimer_2.num_particles)])
        sys._oqs__calc_lindbladian()
        l_actual = sys._oqs__lindbladian
        norm_diff = sps_mtx_norm(l_expected - l_actual)
        self.assertLess(norm_diff, 1.0e-14)

    def test_calc_driving_lindbladians(self):
        with self.assertRaises(ValueError):
            sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
            sys = oqs(sys_size, 0, 1)
            sys._oqs__calc_driving_lindbladians()

        # Dimer: 1
        sys_size = dimer_get_sys_size(self.dimer_1.num_particles)
        hamiltonians = dimer_get_driving_hamiltonias(self.dimer_1.num_particles)
        functions = dimer_get_driving_functions(
            self.dimer_1.drv_type,
            self.dimer_1.drv_ampl,
            self.dimer_1.drv_freq,
            self.dimer_1.drv_phas
        )
        fn = self.dimer_1.get_path() + 'lindbladian_drv_mtx' + self.dimer_1.get_suffix()
        l_expected = load_sparse_matrix(fn, self.dimer_1.sys_size * self.dimer_1.sys_size)
        sys = oqs(sys_size, 1, 1)
        sys.init_driving(hamiltonians, functions)
        sys._oqs__calc_driving_lindbladians()
        l_actual = sys._oqs__driving_lindbladians[0]
        norm_diff = sps_mtx_norm(l_expected - l_actual)
        self.assertLess(norm_diff, 1.0e-14)

        # Dimer: 2
        sys_size = dimer_get_sys_size(self.dimer_2.num_particles)
        hamiltonians = dimer_get_driving_hamiltonias(self.dimer_2.num_particles)
        functions = dimer_get_driving_functions(
            self.dimer_2.drv_type,
            self.dimer_2.drv_ampl,
            self.dimer_2.drv_freq,
            self.dimer_2.drv_phas
        )
        fn = self.dimer_2.get_path() + 'lindbladian_drv_mtx' + self.dimer_2.get_suffix()
        l_expected = load_sparse_matrix(fn, self.dimer_2.sys_size * self.dimer_2.sys_size)
        sys = oqs(sys_size, 1, 1)
        sys.init_driving(hamiltonians, functions)
        sys._oqs__calc_driving_lindbladians()
        l_actual = sys._oqs__driving_lindbladians[0]
        norm_diff = sps_mtx_norm(l_expected - l_actual)
        self.assertLess(norm_diff, 1.0e-14)
