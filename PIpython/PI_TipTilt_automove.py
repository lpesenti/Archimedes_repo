import configparser

import PI_functions  # import the Analysis module

while True:
    config = configparser.ConfigParser()
    config.read('PI_config.ini')

    abort = config.getboolean('DEFAULT', 'abort')
