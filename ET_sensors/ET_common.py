__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti", "Davide Rozza", "ET collaboration"]
__version__ = "0.0.5"
__maintainer__ = "Luca Pesenti (until September 30, 2022)"
__email__ = "lpesenti@uniss.it"
__status__ = "Production"

r"""
In this file are stored all the functions that are not strictly related to the Archimedes experiment. However, they are
used in various methods inside the codes.
"""

# TODO: should we merge this file with Arc_common.py? Maybe it is better to have only one file with 'common' functions

import os

import numpy as np
import pandas as pd


def NLNM(unit):
    """
    The Peterson model represents an ensemble of seismic spectra measured in a worldwide network (Peterson 1993,
    U.S. Geol. Surv. Rept. 2 93–322 https://pubs.er.usgs.gov/publication/ofr93322). In this way it is
    possible to define a low noise model (NLNM) and an high noise model (NHNM) representing respectively the minimum and
    the maximum natural seismic background that can be found on Earth. Here we define these two curves.

    :type unit: int
    :param unit: 1 = displacement, 2 = speed

    :return: A tuple of numpy.ndarray: [NLNM frequencies, NLNM values, NHNM frequencies, NHNM values]
    """
    PL = np.array([0.1, 0.17, 0.4, 0.8, 1.24, 2.4, 4.3, 5, 6, 10, 12, 15.6, 21.9, 31.6, 45, 70,
                   101, 154, 328, 600, 10000])
    AL = np.array([-162.36, -166.7, -170, -166.4, -168.6, -159.98, -141.1, -71.36, -97.26,
                   -132.18, -205.27, -37.65, -114.37, -160.58, -187.5, -216.47, -185,
                   -168.34, -217.43, -258.28, -346.88])
    BL = np.array([5.64, 0, -8.3, 28.9, 52.48, 29.81, 0, -99.77, -66.49, -31.57, 36.16,
                   -104.33, -47.1, -16.28, 0, 15.7, 0, -7.61, 11.9, 26.6, 48.75])

    PH = np.array([0.1, 0.22, 0.32, 0.8, 3.8, 4.6, 6.3, 7.9, 15.4, 20, 354.8, 10000])
    AH = np.array([-108.73, -150.34, -122.31, -116.85, -108.48, -74.66, 0.66, -93.37, 73.54,
                   -151.52, -206.66, -206.66])
    BH = np.array([-17.23, -80.5, -23.87, 32.51, 18.08, -32.95, -127.18, -22.42, -162.98,
                   10.01, 31.63, 31.63])

    fl = 1 / PL
    fh = 1 / PH
    lownoise = 10 ** ((AL + BL * np.log10(PL)) / 20)
    highnoise = 10 ** ((AH + BH * np.log10(PH)) / 20)

    if unit == 1:  # displacement
        lownoise = lownoise * (PL / (2 * np.pi)) ** 2
        highnoise = highnoise * (PH / (2 * np.pi)) ** 2

    if unit == 2:  # speed
        lownoise = lownoise * (PL / (2 * np.pi))
        highnoise = highnoise * (PH / (2 * np.pi))

    return fl, lownoise, fh, highnoise


def check_dir(path, name_dir):
    r"""
    This function checks whether the folder exists in the path. If it exists, it does nothing, while instead it creates
    a new folder in the specific path with the given name.

    :type path: str
    :param path: the path where you want to create the folder

    :type name_dir: str
    :param name_dir: the name of the directory you want to create

    :return: the full path to the new directory
    """
    new_path = path + name_dir
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path


# work in progress ...
def check_df(path, df_name):
    r"""
    This function does something similar to the check_dir(). However, in this case it checks for the existence of a
    DataFrame. If it exists, it does nothing, while instead it creates a new DataFrame in the specific path with the
    specified name. The DataFrame will be created as a 'parquet' file with level 9 'brotli' compression.

    :type path: str
    :param path: the path where you want to create the DataFrame

    :type df_name: str
    :param df_name: the name of the DataFrame you want to create

    :return: the DataFrame found or created
    """
    new_path = path + df_name + '.parquet.brotli'
    if not os.path.exists(new_path):
        df = pd.DataFrame()
        df.to_parquet(new_path, compression='brotli', compression_level=9)
    return pd.read_parquet(new_path)
