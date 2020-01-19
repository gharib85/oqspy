import numpy as np
from scipy.sparse import csr_matrix
import math


def dimer_get_sys_size(num_particles):
    sys_size = num_particles + 1
    return sys_size


def dimer_get_periods(freq):
    period = 2.0 * np.pi / freq
    return [period]


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


def dimer_get_driving_hamiltonias(num_particles):
    sys_size = dimer_get_sys_size(num_particles)

    rows = []
    cols = []
    vals = []

    for st_id in range(0, sys_size):
        rows.append(st_id)
        cols.append(st_id)
        val = -1.0 * float((sys_size - (st_id + 1)) - st_id)
        vals.append(val)

    trace = sum(vals) / float(sys_size)
    vals = [val - trace for val in vals]

    hamiltonians = [csr_matrix((vals, (rows, cols)), shape=(sys_size, sys_size))]
    return hamiltonians


def dimer_get_driving_functions(type, ampl, freq, phas):
    if type == 0:
        period = dimer_get_periods(freq)[0]

        def driving(time):
            mod_time = math.fmod(time, period)
            half_period = period * 0.5
            if mod_time < half_period:
                drv = ampl
            else:
                drv = -ampl
            return drv
    else:
        def driving(time):
            drv = ampl * np.sin(freq * time + phas)
            return drv
    return [driving]
