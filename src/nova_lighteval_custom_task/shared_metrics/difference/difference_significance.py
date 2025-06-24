import numpy as np
import numpy.typing as npt
from scipy import stats


def paired_t_test(diff: npt.ArrayLike, alpha: float = 0.05) -> int:
    """
    Performs a two-sided paired t-test to determine if there is a significant difference
    between paired observations.

    Implementation based on: (doc/references/Significance-Test-for-Deflection-Rate-Difference.pdf)

    :param diff: array_like
        Array of differences between paired observations.
        For example, if comparing before/after measurements,
        arr = [after_1 - before_1, after_2 - before_2, ...]
    :param alpha: float, optional
        Significance level for the test (default is 0.05).
        Must be between 0 and 1.
    :return: int
        1 if the difference between paired observations is significant,
        0 otherwise.
    """
    d = np.asarray(diff, dtype=float)
    n = len(d)
    if n <= 1:
        raise ValueError(f"To perform the paired t-test, the length of input should be greater than 1 but got: {n}")

    d_mean = d.mean()
    d_std = d.std(ddof=1)

    t_stat = d_mean / (d_std / np.sqrt(n))

    # n - 1 degrees of freedom
    df = n - 1

    # Two-sided p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return int(p_value < alpha)
