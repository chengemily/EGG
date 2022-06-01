from scipy import stats
from scipy import integrate

def normality_test(data, alpha=1e-3, verbose=False) -> bool:
    """
    Returns whether it passes normality test.
    """
    stat, p = stats.normaltest(data)
    passes = False
    if verbose: print('Dagostino K2 test')
    if p < alpha:
        if verbose: print('Probably not normal')
    else:
        passes = True
        if verbose: print('Cannot reject that data comes from normal distribution')

    if verbose: print('Shapiro-wilk')
    stat, p = stats.shapiro(data)
    if p < alpha:
        passes = False
        if verbose: print('Probably not normal')
    else:
        if verbose: print('Cannot reject that data comes from normal distribution')

    return passes