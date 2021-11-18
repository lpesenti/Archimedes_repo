import configparser

import OL_functions as of

config = configparser.ConfigParser()
config.read('config.ini')

data_path = config['Paths']['data_dir']
filename_prefix = config['Paths']['prefix_filename']
file = config['Paths']['filename']
quantity = config['Quantities']['qty']

# of.plot_3d_oscilloscope(data_path + file, quantity)
of.plot_3d_crio(data_path, filename_prefix)
# of.eval_mean(data_path + file)
