__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti", "Davide Rozza"]
__version__ = "1.2.1"
__maintainer__ = "Luca Pesenti"
__email__ = "l.pesenti6@campus.unimib.it"
__status__ = "Production"

import datetime
import numpy as np
import pandas as pd
import math


def find_rk(seq, subseq):
    """
    Find the index of the first element of a subsequence.

    Parameters
    ----------
    seq : array_like
        List in which search for the sub-sequence
    subseq : array_like
        Sub-sequence to search for

    Returns
    -------
    out : generator.object
        A generator containing all index where the sub-sequence has been found.

    Examples
    --------
    >>> lst = [1,2,3,4,5,6,4,5,6]
    >>> sub_lst = [4,5,6]
    >>> find_rk(lst, sub_lst)
    <generator object find_rk at 0x0000027C9F6AF9C8>

    >>> list(find_rk(lst, sub_lst))
    [3, 6]

    See Also
    --------
    Additional help can be found in the answer of norok2 `in this discussion
    <https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray>`_.

    """
    n = len(seq)
    m = len(subseq)
    if np.all(seq[:m] == subseq):
        yield 0
    hash_subseq = sum(hash(x) for x in subseq)  # compute hash
    curr_hash = sum(hash(x) for x in seq[:m])  # compute hash
    for i in range(1, n - m + 1):
        curr_hash += hash(seq[i + m - 1]) - hash(seq[i - 1])  # update hash
        if hash_subseq == curr_hash and np.all(seq[i:i + m] == subseq):
            yield i


def find_factors(n):
    """
    Find all the factors of a given number 'n'

    Parameters
    ----------
    n : int
        is the number whose factors you want to know.

    Notes
    -----
    This function return all the factors, not only prime factors.

    Returns
    -------
    out : array_like
        A list containing all the factors in the format [a,b,c,d], where a * b = c * d = n

    Examples
    --------
    >>> find_factors(20)
    array([ 1., 20.,  2., 10.,  4.,  5.])
    """
    factor_lst = np.array([])
    # Note that this loop runs till square root
    i = 1
    while i <= math.sqrt(n):
        if n % i == 0:
            # If divisors are equal, print only one
            if n / i == i:
                factor_lst = np.append(factor_lst, i)
            else:
                # Otherwise print both
                factor_lst = np.append(factor_lst, [i, n / i])
        i = i + 1
    return factor_lst


def from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)


def vectorizer(input_func):
    def output_func(array_of_numbers):
        return [input_func(a) for a in array_of_numbers]

    return output_func


def time_tick_formatter(val, pos=None):
    """
    Return val reformatted as a human readable date

    See Also
    --------
    time_evolution : it is used to rewrite x-axis
    """
    val = str(datetime.datetime.fromtimestamp(val).strftime('%b %d %H:%M:%S'))
    return val


def hex_converter(path_to_data):
    data = pd.read_table(path_to_data, sep=' ', na_filter=False, low_memory=False, engine='c', usecols=[2, 3, 4],
                         header=None, dtype = str, converters={'conv': lambda x: hex(int(str(x), 2))})
    data[[2, 3, 4]] = data[[2, 3, 4]].applymap(lambda x: hex(int(str(x), 2)))
    return data
