__author__ = "Luca Pesenti"
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.2.5"
__maintainer__ = "Luca Pesenti (until September 30, 2022)"
__email__ = "lpesenti@uniss.it"
__status__ = "Prototype"

r"""
[LAST UPDATE: September 7, 2022 - Luca Pesenti]

This file is to be used as a prototype to launch the functions contained in OL_functions.py
However, it is perfectly functioning.
To change some variable, please refer to the OL_config.ini before. Inside this file it is possible to change several
variables without modifying the code directly.
"""

import configparser

import OL_functions as of

# import OL_noise as on

config = configparser.ConfigParser()
config.read('OL_config.ini')

data_path = config['Paths']['data_dir']
filename_prefix = config['Paths']['prefix_filename']
file = config['Paths']['filename']
quantity = config['Quantities']['qty']

if __name__ == '__main__':
    # of.plot_3d_oscilloscope(data_path + file, quantity)
    # of.plot_3d_crio(data_path, filename_prefix)
    # x, y = of.eval_mean(data_path + file)
    # of.linearity(x, y)
    # of.ols_saturation(colorregion=False, limitlines=True)
    of.sled_p_to_i()
