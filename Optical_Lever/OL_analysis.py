import configparser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib import mlab
import OL_functions as of
# import OL_noise as on
import seaborn as sns

config = configparser.ConfigParser()
config.read('config.ini')

data_path = config['Paths']['data_dir']
filename_prefix = config['Paths']['prefix_filename']
file = config['Paths']['filename']
quantity = config['Quantities']['qty']

if __name__ == '__main__':
    # of.plot_3d_oscilloscope(data_path + file, quantity)
    of.plot_3d_crio(data_path, filename_prefix)
    # x, y = of.eval_mean(data_path + file)
    # of.linearity(x, y)
    # of.ols_saturation(colorregion=False, limitlines=True)
    # of.sled_p_to_i()
