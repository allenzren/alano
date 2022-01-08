import numpy as np

def full_jacob_pb(jac_t, jac_r):
    return np.vstack(
        (jac_t[0], jac_t[1], jac_t[2], jac_r[0], jac_r[1], jac_r[2]))