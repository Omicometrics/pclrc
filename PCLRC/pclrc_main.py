"""
This module implements probabilistic context likelihood of relatedness
using Pearson correlation coefficients.

References:
    [1] Saccenti E, et al. J. Proteome Res. 2015, 14, 2, 1101–1111.
    [2] Suarez-Diez M, et al. J. Proteome Res. 2015, 14, 12, 5119–5130.

"""

import tqdm
import numpy as np

from .core import pclrc, pclrc_single


class PCLRC(object):
    """
    This class implements probabilistic context likelihood of relatedness.

    Parameters:
        num_sampling: Number of subsampling.
        frac_sampling: Fraction of samples subsampled.
        q: A number between 0 and 1 to justify the threshold in Pearson
            correlation coefficients to define the associations.
        bootstrap: Whether to use bootstrap for sampling.

    """
    def __init__(self, num_sampling: int = int(1e5),
                 frac_sampling: float = 0.75,
                 q: float = 0.3, bootstrap: bool = False):
        self.num_sampling: int = int(num_sampling)
        self.frac_sampling: float = frac_sampling
        self.q: float = q
        self.bootstrap: bool = bootstrap

        self._check_params()

    def corr_probs(self, x: np.ndarray) -> np.ndarray:
        """
        Computes a probabilistic correlation matrix.

        Parameters
        ----------
        x: Data matrix with n samples in rows by p variables in columns.

        Returns
        -------

        """
        r, c = x.shape
        if r >= 50. or c >= 60.:
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

    def _check_params(self) -> None:
        """ Checks parameters. """
        if not 0.0 <= self.frac_sampling <= 1.0:
            raise ValueError('Frac_sampling must be between 0 and 1., '
                             f'got {self.frac_sampling}.')

        if not 0.0 <= self.q <= 1.0:
            raise ValueError(f'q must be between 0. and 1., got {self.q}.')
