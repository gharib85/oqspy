"""Main module."""


class oqs:

    def __init__(self, sys_size, num_driving_segments, num_dissipators):
        self.__sys_size = sys_size
        self.__num_driving_segments = num_driving_segments
        self.__num_dissipators = num_dissipators

    def init_hamiltonian(self, hamiltonian):
        self.__hamiltonian = hamiltonian
