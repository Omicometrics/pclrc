# pclrc
Probabilistic context likelihood of relatedness<sup>[1]</sup> to study the
correlation network differences<sup>[2]</sup>. This package is developed using Python 3.10.

# Note
The package can be extremely slow for relatively large number of samples or
variables.

# Requirements
The following packages are required:
```
cython 3.X
numpy
matplotlib
tqdm
```
All can be easily installed using `pip` or `conda`.

# Installation
To install, download the package from the
**[Releases](https://github.com/Omicometrics/pclrc/releases/tag/v0.1.0)**.
Uncompress the file and move them to the working directory. Open command line
tool like `Command Prompt` (in Windows) or `Terminal` (in macOS) and set
current directory to the working directory using `cd` command. In Windows, run

```Python -m setup install```

or in macOS, run

```Python3 -m setup install```

The package will be installed.

# Run
To run the package, in `Python IDLE` or other environment like
[JupyterLab](https://jupyter.org),

```from PCLRC import PCLRC```

- Create `PCLRC` object with specified arguments:

  ```pclrc = PCLRC(num_sampling: int = int(1e5), frac_sampling: float = 0.75, q: float = 0.3, corr_prob: float = 0.9, bootstrap: bool = False, num_perms: int = int(1e4), num_cores: Optional[int] = None)```
  * Arguments:
    * `num_sampling`: The number of subsampling procedures. Defaults to
      `100000`.
    * `frac_sampling`: Fraction of samples selected in each subsampling
    procedure to calculate Pearson Correlation Coefficients (PCCs). Defaults
    to 0.75.
    * `q`: The top `q * 100%` PCCs are considered as valid associations using
    the data subsampled, which will be set to 1 in binary adjacent matrix,
    otherwise will be set to 0. Defaults to `0.3`.
      * If `q` is set to `0.`, a hard threshold defined in Ref [3] is used.
    * `corr_prob`: Correlation probability threshold for justifying significant
    associations, so that corresponding PCCs will be used to calculating
    differences in connectivity.
    * `bootstrap`: Whether to use bootstrap sampling during subsampling
    procedures. This is used because some dataset may have very small number
    of samples. Defaults to `False`.
    * `num_perms`: Number of permutations used in permutation tests. Defaults
    to `10000`.
    * `num_cores`: Whether running the permutation tests in parallel and number
    of cores used for the parallelization. Defaults to `None` which means no
    parallelization is used.

- Run `PCLRC` with data matrix `x` and sample labels in `groups`:
  
  ```pclrc.network_diffs(x: np.ndarray, groups: np.ndarray)```
  * Arguments:
    * `x`: Data matrix with size `n` rows by `p` columns, where `n` is the
    number of samples, and `p` is number of variables.
    * `groups`: Group names in 1-D array with number of `n` elements.
  > **[!NOTE]**  
  > Only two groups are allowed for running the analysis, i.e., `len(set(groups))=2`.

- Other than the method `network_diffs`, a method is implemented to obtain the
correlation probability matrix:

  `pclrc.corr_probs(x: np.ndarray, prog_bar: bool = True)`
  * Arguments:
    * `x`: Data matrix with size `n` rows by `p` columns, where `n` is the
    number of samples, and `p` is number of variables.
    * `prog_bar`: Whether to show progress bar for calculation.

# Analyze the results

After running `pclrc.network_diffs`,

- to obtain the correlation probability:
  `pclrc.pearson_corr_probs(label: Optional[Any] = None)`
  * Arguments:
    * `label`: Group name in `groups` input when running `pclrc.network_diffs`,
    if set it to `None`, correlation probability for both groups will be
    output, with first prob. matrix corresponding to group `groups[0]`, second
    to group `groups[1]`.
- to obtain differences in connectivity:
  `pclrc.diff_connectivity`
- to obtain the significant differences in connectivity:
  `pclrc.sig_diff_connectivity(fdr: float=0.05)`

  * Arguments:
    * `fdr`: The false discovery rate (FDR) to select significant differences
    in connectivity, which is the adjust *p* value threshold after BH
    correction.
- to obtain permutated probabilities for all variables: ```pclrc.perm_probs```
- to obtain the labels/group names: `pclrc.data_labels`

# Demonstration
To demonstrate how to run the package, a jupyter notebook was provided in the 
package, which can also be obtained at
[**<ins>pclrc.demo.ipynb</ins>**](https://github.com/Omicometrics/pclrc/blob/main/pclrc_demo.ipynb).
The dataset was downloaded from [Metabolomics Workbench](https://www.metabolomicsworkbench.org),
study [ST003751](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST003751&StudyType=MS&ResultType=1),
in negative mode. The figures shown in the references, e.g., connections
between variables, differences in connectivity, were also provided in the
demonstration.


# References
[1] Saccenti E, et al. J. Proteome Res. 2015, 14, 2, 1101–1111.<br>
[2] Vignoli A, et al. J. Proteome Res. 2020, 19, 949−961.<br>
[3] Suarez-Diez M, et al. J. Proteome Res. 2015, 14, 12, 5119–5130.
