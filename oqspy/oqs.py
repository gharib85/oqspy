from scipy.sparse import csr_matrix
import types
from inspect import signature


class oqs:

    def __init__(self, sys_size, num_driving_segments, num_dissipators):
        """
         Open Quantum System (OQS) basic initialization

        :param sys_size:
            Number of states in OQS.
        :type sys_size: int

        :param num_driving_segments:
            Number of driving segments in OQS.
            Must be 0 if OQS is autonomous.
        :type num_driving_segments: int

        :param num_dissipators:
            Number of dissipators in OQS.
        :type num_dissipators: int
        """
        if not isinstance(sys_size, int):
            raise TypeError('sys_size must be integer.')
        if sys_size <= 0:
            raise ValueError('sys_size must be positive integer.')

        if not isinstance(num_driving_segments, int):
            raise TypeError('num_driving_segments must be integer.')
        if num_driving_segments < 0:
            raise ValueError('num_driving_segments must be non-negative integer.')

        if not isinstance(num_dissipators, int):
            raise TypeError('num_dissipators must be integer.')
        if num_dissipators < 0:
            raise ValueError('num_dissipators must be positive integer.')

        self.__sys_size = sys_size
        self.__num_driving_segments = num_driving_segments
        self.__num_dissipators = num_dissipators

        self.__hamiltonian = None
        self.__driving_hamiltonians = None
        self.__driving_functions = None
        self.__dissipators = None

    def init_hamiltonian(self, hamiltonian):
        """
        Initialization of Open Quantum System (OQS) with Hamiltonian.

        :param hamiltonian:
            Hamiltonian CSR matrix.
        :type hamiltonian: csr_matrix
        """
        if not isinstance(hamiltonian, csr_matrix):
            raise TypeError('hamiltonian must be csr_matrix.')
        if hamiltonian.shape[0] != self.__sys_size or hamiltonian.shape[1] != self.__sys_size:
            raise ValueError('Incorrect size of hamiltonian.')
        self.__hamiltonian = hamiltonian

    def init_driving(self, hamiltonias, functions):
        """
        Initialization of Open Quantum System (OQS) with driving details.

        :param hamiltonias:
            List of driving Hamiltonians (CSR format).
        :type hamiltonias: list

        :param functions:
            List of driving functions.
        :type functions: list
        """
        if not isinstance(hamiltonias, list):
            raise TypeError('Driving hamiltonians must be list of csr_matrix.')
        if not isinstance(functions, list):
            raise TypeError('Driving functions must be list of functions.')

        if not hamiltonias:
            if self.__num_driving_segments > 0:
                raise ValueError('Wrong number of driving hamiltonians.')
        else:
            if len(hamiltonias) != self.__num_driving_segments:
                raise ValueError('Wrong number of driving hamiltonians.')
            if not all(isinstance(x, csr_matrix) for x in hamiltonias):
                raise TypeError('Driving hamiltonians must be list of csr_matrix.')
            for h in hamiltonias:
                if h.shape[0] != self.__sys_size or h.shape[1] != self.__sys_size:
                    raise ValueError('Incorrect size of driving hamiltonians.')

        if not functions:
            if self.__num_driving_segments > 0:
                raise ValueError('Wrong number of driving functions.')
        else:
            if len(functions) != self.__num_driving_segments:
                raise ValueError('Wrong number of driving functions.')
            if not all(isinstance(x, types.FunctionType) for x in functions):
                raise TypeError('Driving functions must be list of functions.')
            for f in functions:
                sig = signature(f)
                if len(sig.parameters) != 1:
                    raise ValueError('Driving functions must have only one input argument (time).')

        self.__driving_hamiltonians = hamiltonias
        self.__driving_functions = functions

    def init_dissipators(self, dissipators):
        """
        Initialization of Open Quantum System (OQS) with dissipators (list of CSR matrices).

        :param dissipators:
            List of dissipators (CSR format).
        :type dissipators: list
        """
        if not isinstance(dissipators, list):
            raise TypeError('dissipators must be list of csr_matrix.')

        if not dissipators:
            raise ValueError('Quantum system is OPEN. There must be at least one dissipator.')
        else:
            if len(dissipators) != self.__num_dissipators:
                raise ValueError('Wrong number of dissipators.')
            if not all(isinstance(x, csr_matrix) for x in dissipators):
                raise TypeError('dissipators must be list of csr_matrix.')
            for d in dissipators:
                if d.shape[0] != self.__sys_size or d.shape[1] != self.__sys_size:
                    raise ValueError('Incorrect size of dissipators.')

        self.__dissipators = dissipators
