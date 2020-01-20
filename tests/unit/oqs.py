import unittest
from oqspy.oqs import oqs
from scipy.sparse import csr_matrix
import numpy as np


class TestOQS(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_oqs_init(self):
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

    def test_oqs_init_hamiltonian(self):
        sys = oqs(10, 1, 1)
        with self.assertRaises(TypeError):
            sys.init_hamiltonian('aaa')

        rows = np.array([0, 0, 1, 2, 2, 2])
        cols = np.array([0, 2, 2, 0, 1, 2])
        vals = np.array([1, 2, 3, 4, 5, 6])
        h = csr_matrix((vals, (rows, cols)), shape=(3, 3))
        with self.assertRaises(ValueError):
            sys.init_hamiltonian(h)

    def test_oqs_init_driving_hamiltonians(self):
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
