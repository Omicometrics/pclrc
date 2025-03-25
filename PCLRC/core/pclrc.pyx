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
cdef void normalize(float[:, ::1] x, float[:, ::1] xt):
    """
    Normalizes x for Pearson correlation coefficients and transposes
    to the matrix with columns as rows of original matrix.
    
    """
    cdef:
        Py_ssize_t i, j
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
            xt[i, j] = (x[j, i] - s) / b


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void clr(float[:, ::1] x, float q, float[:, ::1] corr_x,
              float[:, ::1] b_adjx, float[::1] corr_counts, float[:, ::1] xt):
    """
    Generates an association matrix.
    
    Parameters
    ----------
    x: Data matrix with n samples by p variables.
    q: Threshold % for defining association.
    corr_x: Correlation matrix.
    b_adjx: Binary adjacent matrix.
    corr_counts: 1-D array to count the number of correlation
        coefficients in each of total 1000 bins.

    Returns
    -------

    """
    cdef:
        Py_ssize_t i, j, t
        Py_ssize_t n = x.shape[0]
        Py_ssize_t p = x.shape[1]
        float fn = <float> n
        float pp = <float> (p * p)
        float s, s2, thr, b

    normalize(x, xt)

    # Pearson correlation
    for i in range(p):
        for j in range(p):
            s = 0.
            for k in range(n):
                s += xt[i, k] * xt[j, k]
            corr_x[i, j] = s
            t = <ssize_t> (fabs(s) * 1000.)
            corr_counts[t] += 1.

    # CLR
    if q != 0.:
        # threshold
        b = 0.
        thr = 1.
        for i in range(999, -1, -1):
            b += corr_counts[i]
            if b / pp > q:
                break
            thr = <float> (i + 1) / 1000.

        for i in range(p):
            for j in range(p):
                if fabs(corr_x[i, j]) >= thr:
                    b_adjx[i, j] = 1.

        return


    cdef:
        float * norm_x = <float *> malloc(p * sizeof(float))
        float * mx = <float *> malloc(p * sizeof(float))

    pp = <float> p
    # hard threshold
    for i in range(p):
        s = 0.
        s2 = 0.
        for j in range(p):
            s2 += corr_x[i, j] * corr_x[i, j]
            s += corr_x[i, j]
        mx[i] = s / pp
        norm_x[i] = sqrtf(s2 / pp - mx[i] * mx[i])

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
        float[::1] corr_counts = np.zeros(1000, dtype=DTYPE_F)
        float[:, ::1] xt = np.zeros((p, s), dtype=DTYPE_F)
        float[:, ::1] corr_x = np.zeros((p, p), dtype=DTYPE_F)
        float[:, ::1] sub_x = np.zeros((s, p), dtype=DTYPE_F)
        float[:, ::1] b_adjx = np.zeros((p, p), dtype=DTYPE_F)

    # subsampling
    for i in range(s):
        sub_x[i, :] = x[rnd_ix[i]]

    # CLR
    clr(sub_x, q, corr_x, b_adjx, corr_counts, xt)

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
        float[::1] corr_counts = np.zeros(1000, dtype=DTYPE_F)
        float[:, ::1] xt = np.zeros((p, s), dtype=DTYPE_F)
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
        clr(sub_x, q, corr_x, b_adjx, corr_counts, xt)

        # get probabilities
        for j in range(p):
            for k in range(p):
                prob_adjs[j, k] += b_adjx[j, k]

        b_adjx[:, :] = 0.
        corr_counts[:] = 0.

    # get the probability
    for i in range(p):
        for j in range(p):
            prob_adjs[i, j] /= fr

    return np.asarray(prob_adjs)
