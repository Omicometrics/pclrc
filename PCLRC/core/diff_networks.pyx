cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrtf
from .pclrc cimport clr

import numpy as np
cimport numpy as np

np.import_array()

DTYPE_F = np.float32
DTYPE_I = np.int32


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void chi_conn(float[:, ::1] x,  float[:, ::1] probs, float prob_thr,
                   float[::1] chi_vals):
    """
    Calculates Pearson correlation coefficients and chi stats.

    """

    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        float fn = <float> n
        float s, s2, b

    for i in range(p):
        s = 0.
        s2 = 0.
        for j in range(n):
            s += x[j, i]
            s2 += x[j, i] * x[j, i]
        s /= fn
        b = sqrtf(s2 - fn * s * s)
        # save to a transposed data matrix
        for j in range(n):
            x[j, i] = (x[j, i] - s) / b

    # Pearson correlation
    for i in range(p):
        b = 0.
        for j in range(p):
            if probs[i, j] >= prob_thr:
                s = 0.
                for k in range(n):
                    s += x[k, i] * x[k, j]
                b += fabs(s)
        chi_vals[i] = b - 1.


@cython.boundscheck(False)
@cython.wraparound(False)
def diff_connect(float[:, ::1] probs_a, float[:, ::1] x_a,
                 float[:, ::1] probs_b, float[:, ::1] x_b, float prob_thr):
    """
    Calculates differential connectivity.

    Parameters
    ----------
    probs_a: PCLRC probability matrix for network a.
    x_a: Data matrix for generating network a, with na samples and p
        variables.
    probs_b: PCLRC probability matrix for network b.
    x_b: Data matrix for generating network b, with nb samples and p
        variables.
    prob_thr: Probability threshold.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i
        Py_ssize_t p = x_a.shape[1]
        float[::1] chi_a = np.zeros(p, dtype=DTYPE_F)
        float[::1] chi_b = np.zeros(p, dtype=DTYPE_F)
        float[::1] diff_chis = np.zeros(p, dtype=DTYPE_F)

    chi_conn(x_a, probs_a, prob_thr, chi_a)
    chi_conn(x_b, probs_b, prob_thr, chi_b)

    for i in range(p):
        diff_chis[i] = chi_a[i] - chi_b[i]

    return np.asarray(diff_chis)


@cython.boundscheck(False)
@cython.wraparound(False)
def perm_diff_connect(float[:, ::1] x, int[::1] group_tags,
                      int r, float f, float q, int bootstrap, float prob_thr):
    """
    Performs permutation tests to get the differential connectivity.

    Parameters
    ----------
    x: Data matrix for analysis, with size of n by p, where n is the
        number of samples, p is the number variables/molecules.
    group_tags: Tags for groups, 0 for group 1, 1 for group 2.
    r: Number of random sampling.
    f: Fraction randomly sampled.
    q: Threshold for defining the association of molecules.
        0.: a hard threshold will be used.
        any value between 0. and 1., the top q * 100.% will be used.
    bootstrap: Whether to use bootstrap.
    prob_thr: Probability threshold.

    Returns
    -------

    """

    cdef:
        Py_ssize_t i, j, k, g, m, kn, n1
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        int[::1] rnd_ix = np.zeros(n, dtype=DTYPE_I)
        float[:, ::1] rnd_x = np.zeros((n, p), dtype=DTYPE_F)
        float[:, ::1] xt = np.zeros((p, n), dtype=DTYPE_F)
        float[:, ::1] corr_x = np.zeros((p, p), dtype=DTYPE_F)
        float[:, ::1] sub_x = np.zeros((n, p), dtype=DTYPE_F)
        float[:, ::1] b_adjx = np.zeros((p, p), dtype=DTYPE_F)
        float[:, ::1] tmp_probs = np.zeros((p, p), dtype=DTYPE_F)
        float[:, ::1] chis = np.zeros((2, p), dtype=DTYPE_F)
        float[::1] corr_counts = np.zeros(1001, dtype=DTYPE_F)
        float[::1] diff_chis = np.zeros(p, dtype=DTYPE_F)
        float fr = <float> r

    for g in range(2):
        # randomize each column
        n1 = 0
        for j in range(n):
            if group_tags[j] == g:
                n1 += 1

        for j in range(p):
            rnd_ix.base[:n1] = np.ascontiguousarray(np.random.permutation(n1),
                                                    dtype=DTYPE_I)
            for i in range(n1):
                rnd_x[i, j] = x[rnd_ix[i], j]

        # PCLRC
        kn = <ssize_t> (<float> n1 * f)
        for i in range(r):
            if bootstrap == 1:
                rnd_ix.base[:kn] = np.ascontiguousarray(
                    np.random.choice(n1, size=kn, replace=True),
                    dtype=DTYPE_I
                )
            else:
                rnd_ix.base[:kn] = np.ascontiguousarray(
                    np.random.choice(n1, size=kn, replace=False),
                    dtype=DTYPE_I
                )

            # subsampling
            for j in range(kn):
                sub_x[j, :] = rnd_x[rnd_ix[j]]

            # CLR
            clr(sub_x, q, corr_x, b_adjx, corr_counts, xt)

            # sum up adjacent matrix
            for j in range(p):
                for k in range(p):
                    tmp_probs[j, k] += b_adjx[j, k]

            b_adjx[:, :] = 0.
            corr_counts[:] = 0.

        # get the probability
        for i in range(p):
            for j in range(p):
                tmp_probs[i, j] /= fr

        # connectivity
        chi_conn(rnd_x[:n1], tmp_probs, prob_thr, chis[g])
        tmp_probs[:, :] = 0.

    for i in range(p):
        diff_chis[i] = chis[0, i] - chis[1, i]

    return np.asarray(diff_chis)
