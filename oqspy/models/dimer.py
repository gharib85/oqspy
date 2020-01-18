import numpy as np
from scipy.sparse import csr_matrix


def dimer_get_sys_size(num_particles):
    sys_size = num_particles + 1
    return sys_size


def dimer_get_hamiltonian(num_particles, E, U, J):
    sys_size = dimer_get_sys_size(num_particles)
    U /= float(num_particles)

    rows = []
    cols = []
    vals = []

    for st_id in range(0, sys_size):
        rows.append(st_id)
        cols.append(st_id)
        val_U = 2.0 * U * float(st_id * (st_id - 1) + (sys_size - (st_id + 1)) * (sys_size - (st_id + 1) - 1))
        val_E = -E * float((sys_size - (st_id + 1)) - st_id)
        vals.append(val_U + val_E)

    trace = sum(vals) / float(sys_size)
    vals = [val - trace for val in vals]

    for st_id in range(0, sys_size - 1):
        rows.append(st_id + 1)
        cols.append(st_id)
        vals.append(-J * np.sqrt(float((sys_size - (st_id + 1)) * (st_id + 1))))

        rows.append(st_id)
        cols.append(st_id + 1)
        vals.append(-J * np.sqrt(float((st_id + 1) * (sys_size - (st_id + 1)))))

    hamiltonian = csr_matrix((vals, (rows, cols)), shape=(sys_size, sys_size))

    return hamiltonian
