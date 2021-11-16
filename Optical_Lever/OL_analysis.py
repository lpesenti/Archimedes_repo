import configparser

import OL_functions as of

config = configparser.ConfigParser()
config.read('config.ini')

data_path = config['Paths']['data_dir']
file = config['Paths']['filename']
quantity = config['Quantities']['qty']

# of.plot_3d_ol(data_path + file, quantity)
of.eval_mean(data_path + file)
