"""
This module implements probabilistic context likelihood of relatedness
using Pearson correlation coefficients.

References:
    [1] Saccenti E, et al. J. Proteome Res. 2015, 14, 2, 1101–1111.
    [2] Suarez-Diez M, et al. J. Proteome Res. 2015, 14, 12, 5119–5130.
    [3] Vignoli A, et al. J. Proteome Res. 2020, 19, 949−961.

"""

import tqdm
import numpy as np

from .core import pclrc, pclrc_single, diff_connect, perm_diff_connect


class PCLRC(object):
    """
    This class implements probabilistic context likelihood of relatedness.

    Parameters:
        num_sampling: Number of subsampling.
        frac_sampling: Fraction of samples subsampled.
        q: A number between 0 and 1 to justify the threshold in Pearson
            correlation coefficients to define the associations.
        prob: Probability threshold to define the associations.
        bootstrap: Whether to use bootstrap for sampling.
        num_perms: Number of permutations for testing the significance.

    """
    def __init__(self, num_sampling: int = int(1e5),
                 frac_sampling: float = 0.75,
                 q: float = 0.3, prob: float = 0.9,
                 bootstrap: bool = False, num_perms: int = int(1e4)):
        self.num_sampling: int = int(num_sampling)
        self.frac_sampling: float = frac_sampling
        self.q: float = q
        self.bootstrap: bool = bootstrap
        self.prob: float = prob
        self.num_perms: int = int(num_perms)

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

        # permutation tests
        xc: np.ndarray = np.ascontiguousarray(x, dtype=np.float32)
        p: int = d_chis.size
        g_tags: np.ndarray = np.ones(x.shape[0], dtype=np.int32)
        g_tags[ix] = 0
        rnd_d_chis = np.zeros((self.num_perms, p), dtype=np.float32)
        for i in tqdm.tqdm(range(self.num_perms), desc="Permutation tests"):
            rnd_d_chis[i, :] = perm_diff_connect(
                xc, g_tags, self.num_sampling, self.frac_sampling, self.q,
                int(self.bootstrap), self.prob
            )

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
