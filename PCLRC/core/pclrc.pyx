cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport fabs, sqrtf

import numpy as np
cimport numpy as np

np.import_array()


DTYPE_I = np.int32
DTYPE_F = np.float32


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void clr(float[:, ::1] x, float q, float[:, ::1] corr_x,
              float[:, ::1] b_adjx, float[::1] corr_xvals):
    """
    Generates an association matrix.
    
    Parameters
    ----------
    x: Data matrix with n samples by p variables.
    q: Threshold % for defining association.
    corr_x: Correlation matrix.
    b_adjx: Binary adjacent matrix.
    corr_xvals: 1-D correlation array for sorting.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j, k, t
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        Py_ssize_t m = 0
        int[::1] ix
        float * norm_x = <float *> malloc(p * sizeof(float))
        float * mx = <float *> malloc(p * sizeof(float))
        float fn = <float> n
        float s, s2, thr

    for i in range(p):
        s = 0.
        s2 = 0.
        for j in range(n):
            s += x[j, i]
            s2 += x[j, i] * x[j, i]
        mx[i] = s / fn
        norm_x[i] = sqrtf(s2 - s * s / fn)

    # Pearson correlation
    for i in range(p):
        for j in range(p):
            s = 0.
            for k in range(n):
                s += x[k, i] * x[k, j]
            corr_x[i, j] = (s - fn * mx[i] * mx[j]) / (norm_x[i] * norm_x[j])
            corr_xvals[m] = fabs(corr_x[i, j])
            m += 1

    # CLR
    if q != 0.:
        ix = np.ascontiguousarray(np.argsort(corr_xvals), dtype=DTYPE_I)
        t = <ssize_t> ix[<ssize_t> ((1. - q) * <float> (p * p))]

        # threshold
        thr = corr_xvals[t]
        for i in range(p):
            for j in range(p):
                if fabs(corr_x[i, j]) >= thr:
                    b_adjx[i, j] = 1.

    else:
        # hard threshold
        for i in range(p):
            s = 0.
            s2 = 0.
            for j in range(p):
                s2 += corr_x[i, j] * corr_x[i, j]
                s += corr_x[i, j]
            mx[i] = s / <float> p
            norm_x[i] = sqrtf(s2 / <float> p - mx[i] * mx[i])

        for i in range(p):
            for j in range(p):
                # z score
                s = max((corr_x[i, j] - mx[i]) / norm_x[i], 0.)
                s2 = max((corr_x[i, j] - mx[j]) / norm_x[j], 0.)
                if s * s + s2 * s2 > 0.:
                    b_adjx[i, j] = 1.

    free(mx)
    free(norm_x)


@cython.boundscheck(False)
@cython.wraparound(False)
def pclrc_single(float[:, ::1] x, float f, float q, int bootstrap,
                 float[:, ::1] prob_adjs):
    """
    Probabilistic context likelihood of relatedness.

    Parameters
    ----------
    x: Data matrix for analysis, with size of n by p, where n is the
        number of samples, p is the number variables/molecules.
    f: Fraction randomly sampled.
    q: Threshold for defining the association of molecules.
        0.: a hard threshold will be used.
        any value between 0. and 1., the top q * 100.% will be used.
    bootstrap: Whether to use bootstrap.
    prob_adjs: Probabilistic adjacent matrix.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j
        Py_ssize_t n = x.shape[0]
        Py_ssize_t s = <ssize_t> (f * <float> n)
        Py_ssize_t p = x.shape[1]
        int[::1] rnd_ix = np.ascontiguousarray(
            np.random.choice(n, size=s, replace=bool(bootstrap)),
            dtype=DTYPE_I)
        float[::1] corr_vals = np.zeros(p * p, dtype=DTYPE_F)
        float[:, ::1] corr_x = np.zeros((p, p), dtype=DTYPE_F)
        float[:, ::1] sub_x = np.zeros((s, p), dtype=DTYPE_F)
        float[:, ::1] b_adjx = np.zeros((p, p), dtype=DTYPE_F)

    # subsampling
    for i in range(s):
        sub_x[i, :] = x[rnd_ix[i]]

    # CLR
    clr(sub_x, q, corr_x, b_adjx, corr_vals)

    # get probabilities
    for i in range(p):
        for j in range(p):
            prob_adjs[i, j] += b_adjx[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
def pclrc(float[:, ::1] x, int r, float f, float q, int bootstrap):
    """
    Probabilistic context likelihood of relatedness.

    Parameters
    ----------
    x: Data matrix for analysis, with size of n by p, where n is the
        number of samples, p is the number variables/molecules.
    r: Number of random sampling.
    f: Fraction randomly sampled.
    q: Threshold for defining the association of molecules.
        0.: a hard threshold will be used.
        any value between 0. and 1., the top q * 100.% will be used.
    bootstrap: Whether to use bootstrap.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t n = x.shape[0]
        Py_ssize_t s = <ssize_t> (f * <float> n)
        Py_ssize_t p = x.shape[1]
        int[::1] rnd_ix = np.zeros(s, dtype=DTYPE_I)
        float[::1] corr_vals = np.zeros(p * p, dtype=DTYPE_F)
        float[:, ::1] corr_x = np.zeros((p, p), dtype=DTYPE_F)
        float[:, ::1] sub_x = np.zeros((s, p), dtype=DTYPE_F)
        float[:, ::1] b_adjx = np.zeros((p, p), dtype=DTYPE_F)
        float[:, ::1] prob_adjs = np.zeros((p, p), dtype=DTYPE_F)
        float fr = <float> r

    for i in range(r):
        if bootstrap == 1:
            rnd_ix.base[:] = np.ascontiguousarray(
                np.random.choice(n, size=s, replace=True),
                dtype=DTYPE_I
            )
        else:
            rnd_ix.base[:] = np.ascontiguousarray(
                np.random.choice(n, size=s, replace=False),
                dtype=DTYPE_I
            )
        # subsampling
        for j in range(s):
            sub_x[j, :] = x[rnd_ix[j]]

        # CLR
        clr(sub_x, q, corr_x, b_adjx, corr_vals)

        # get probabilities
        for j in range(p):
            for k in range(p):
                prob_adjs[j, k] += b_adjx[j, k]

        b_adjx[:, :] = 0.
        corr_vals[:] = 0.

    # get the probability
    for i in range(p):
        for j in range(p):
            prob_adjs[i, j] /= fr

    return np.asarray(prob_adjs)
