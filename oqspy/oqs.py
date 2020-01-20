from scipy.sparse import csr_matrix


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
