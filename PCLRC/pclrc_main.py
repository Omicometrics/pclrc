"""
This module implements probabilistic context likelihood of relatedness
using Pearson correlation coefficients.

References:
    [1] Saccenti E, et al. J. Proteome Res. 2015, 14, 2, 1101–1111.
    [2] Suarez-Diez M, et al. J. Proteome Res. 2015, 14, 12, 5119–5130.
    [3] Vignoli A, et al. J. Proteome Res. 2020, 19, 949−961.

"""

import tqdm
import functools
import numpy as np

from typing import Optional, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from .core import pclrc, pclrc_single, diff_connect, perm_diff_connect, perm_probs


def perm_tests(num_sampling: int, frac_sampling: float, q: float,
               bootstrap: bool, prob_thr: float, x: np.ndarray,
               group_tags: np.ndarray, num_perms: int, k: int) -> np.ndarray:
    """
    Performs permutation tests to calculate the differential
    connectivities.

    Parameters
    ----------
    num_sampling: Number of sampling.
    frac_sampling: Fraction of samples randomly selected for creating
        partial networks.
    q: Percentage of top absolute correlation coefficients considered
        as valid connections.
    bootstrap: Whether to use bootstrap sampling or not.
    prob_thr: Probability threshold.
    x: Data matrix.
    group_tags: Group tags.
    num_perms: Number of permutations.
    k: An integer.

    Returns
    -------

    """
    p: int = x.shape[1]
    b: int = int(bootstrap)
    diff_chis: np.ndarray = np.zeros((num_perms, p), dtype=np.float32)
    for i in tqdm.tqdm(range(num_perms),
                       desc=f"Subprocess {k}: Permutation tests"):
        diff_chis[i, :] = perm_diff_connect(
            x, group_tags, num_sampling, frac_sampling, q, b, prob_thr
        )

    return diff_chis


class PCLRC(object):
    """
    This class implements probabilistic context likelihood of relatedness.

    Parameters:
        num_sampling: Number of subsampling.
        frac_sampling: Fraction of samples subsampled.
        q: A number between 0 and 1 to justify the threshold in Pearson
            correlation coefficients to define the associations.
        corr_prob: Probability threshold to define the associations.
        alpha: Significance level, p value threshold to define the
            significance of the network differentiations, defaults to 0.05.
        bootstrap: Whether to use bootstrap for sampling.
        num_perms: Number of permutations for testing the significance.
        num_cores: Number of cores for running the permutation test in
            parallel.

    """
    def __init__(self, num_sampling: int = int(1e5),
                 frac_sampling: float = 0.75,
                 q: float = 0.3, corr_prob: float = 0.9,
                 alpha: float = 0.05,
                 bootstrap: bool = False, num_perms: int = int(1e4),
                 num_cores: Optional[int] = None):

        self.num_sampling: int = int(num_sampling)
        self.frac_sampling: float = frac_sampling
        self.q: float = q
        self.bootstrap: bool = bootstrap
        self.prob: float = corr_prob
        self.num_perms: int = int(num_perms)
        self.num_cores: Optional[int] = num_cores
        self._sub_nums: Optional[List[int]] = None

        # calculations
        self._labels: Optional[np.ndarray] = None
        self._g1_probs: Optional[np.ndarray] = None
        self._g2_probs: Optional[np.ndarray] = None
        self._diff_chis: Optional[np.ndarray] = None
        self._perm_diff_chis: Optional[np.ndarray] = None

        self._check_params()

    def corr_probs(self, x: np.ndarray, prog_bar: bool = True) -> np.ndarray:
        """
        Computes a probabilistic correlation matrix.

        Parameters
        ----------
        x: Data matrix with n samples in rows by p variables in columns.
        prog_bar: Whether to show progress bar or not.

        Returns
        -------

        """
        if prog_bar:
            r, c = x.shape
            probs = np.zeros((c, c), dtype=np.float32)
            for _ in tqdm.tqdm(range(self.num_sampling),
                               desc='Calculating probs'):
                pclrc_single(np.ascontiguousarray(x, dtype=np.float32),
                             self.frac_sampling, self.q, self.bootstrap, probs)
            probs /= np.float32(self.num_sampling)
        else:
            probs = pclrc(np.ascontiguousarray(x, dtype=np.float32),
                          self.num_sampling, self.frac_sampling, self.q,
                          int(self.bootstrap))
        return probs

    def network_diffs(self, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        Performs network difference analysis

        Parameters
        ----------
        x: Data matrix with n samples in rows by p variables in columns.
        groups: Data groups.

        Returns
        -------

        """
        labels = self._check_groups(groups)
        # get connectivity
        ix = groups == labels[0]
        xa = np.ascontiguousarray(x[ix, :], dtype=np.float32)
        # probability of connectivity for label a
        pb_a = self.corr_probs(xa, prog_bar=False)

        # probability of connectivity for label b
        xb = np.ascontiguousarray(x[~ix, :], dtype=np.float32)
        pb_b = self.corr_probs(xb, prog_bar=False)

        # differential connectivity
        d_chis = diff_connect(pb_a, xa, pb_b, xb, self.prob)

        # assign to attributes
        self._labels = labels
        self._g1_probs = pb_a
        self._g2_probs = pb_b
        self._diff_chis = d_chis

        if self.num_perms == 0:
            return

        # permutation tests
        xc: np.ndarray = np.ascontiguousarray(x, dtype=np.float32)
        p: int = d_chis.size
        g_tags: np.ndarray = np.ones(x.shape[0], dtype=np.int32)
        g_tags[ix] = 0
        rnd_d_chis = np.zeros((self.num_perms, p), dtype=np.float32)
        if self.num_cores is None or self.num_cores == 1:
            # not perform parallelization
            for i in tqdm.tqdm(range(self.num_perms), desc="Permutation tests"):
                rnd_d_chis[i, :] = perm_diff_connect(
                    xc, g_tags, self.num_sampling, self.frac_sampling, self.q,
                    int(self.bootstrap), self.prob
                )
        else:
            perm_tester = functools.partial(
                perm_tests,
                self.num_sampling, self.frac_sampling, self.q, self.bootstrap,
                self.prob, xc, g_tags
            )

            self._get_num_perms()
            # perform permutation tests in parallel
            sub_chis: List[np.ndarray] = []
            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                results = [executor.submit(perm_tester, nk, i + 1)
                           for i, nk in enumerate(self._sub_nums)]
                for f in as_completed(results):
                    sub_chis.append(f.result())

            # combine all arrays
            k: int = 0
            for arr in sub_chis:
                rnd_d_chis[k: k + arr.shape[0], :] = arr
                k += arr.shape[0]

        self._perm_diff_chis = rnd_d_chis

    @property
    def pearson_corr_probs(self, label: Optional[Any] = None)\
            -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Returns PCLRC pearson correlation matrix.

        Parameters:
        ----------
        label: Group name of samples.

        """
        if self._labels[0] == label:
            return self._g1_probs

        if self._labels[1] == label:
            return self._g2_probs

        if label is None:
            return self._g1_probs, self._g2_probs

        raise ValueError(f"Unknown label name input: {label}.")

    @property
    def diff_connectivity(self) -> np.ndarray:
        """ Returns differential connectivity. """
        return self._diff_chis

    @property
    def sig_diff_connectivity(self, fdr: float = 0.05)\
            -> tuple[np.ndarray, np.ndarray]:
        """
        Returns significant differential connectivity with the control
        of FDR, i.e., by Benjamini−Hochberg correction.

        Parameters
        ----------
        FDR: FDR threshold, defaults to 0.05.

        Returns
        -------
        ix: Indexes of variables that are significant.
        pvals: p values.

        """
        if self._perm_diff_chis is None:
            raise ValueError("No permutation tests performed.")

        return perm_probs(self._diff_chis, self._perm_diff_chis,
                          np.float32(fdr))

    def _get_num_perms(self) -> None:
        """
        Gets arrays for saving chi values obtained from permutation
        tests.

        Parameters
        ----------
        p: Number of columns.

        """
        nsub: int = self.num_perms // self.num_cores
        nrest: int = self.num_perms % self.num_cores
        n_subs: List[int] = []
        if nrest > 0:
            n_subs += [nsub + 1 for i in range(nrest)]

        if nsub > 0:
            n_subs += [nsub for i in range(nrest, self.num_cores)]

        self._sub_nums = n_subs

    def _check_params(self) -> None:
        """ Checks parameters. """
        if not 0.0 <= self.frac_sampling <= 1.0:
            raise ValueError('Frac_sampling must be between 0 and 1., '
                             f'got {self.frac_sampling}.')

        if not 0.0 <= self.q <= 1.0:
            raise ValueError(f'q must be between 0. and 1., got {self.q}.')

    def _check_groups(self, y: np.ndarray) -> np.ndarray:
        """ Checks groups in y. """
        groups = np.unique(y)
        if groups.size == 1:
            raise ValueError("Only a single group was found, "
                             "quit the differential analysis.")

        if groups.size > 2:
            raise ValueError("More than two groups were found, "
                             "not applicable using PCLRC.")

        return groups
