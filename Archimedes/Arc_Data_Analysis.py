__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti"]
__version__ = "1.0.0"
__maintainer__ = "Luca Pesenti"
__email__ = "l.pesenti6@campus.unimib.it"
__status__ = "Prototype"

import matplotlib.pyplot as plt
import configparser
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import Arc_functions as af
import numpy as np
import timeit, time
import os
import logging
import logging.handlers
import pickle as pkl

config = configparser.ConfigParser()
config.read('config.ini')
mailing_list = [x for x in config['DEFAULT']['mail_list'].split(',')]

rootLogger = logging.getLogger('data_analysis')
rootLogger.propagate = False
rootLogger.setLevel(logging.DEBUG)
output_formatter = logging.Formatter(
    "[%(asctime)s | %(filename)s %(funcName)15s(), line %(lineno)d] %(levelname)s: %(message)s", '%d-%m-%y %H:%M:%S')
stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
mail_formatter = logging.Formatter(
    "Process ID - name: %(process)d - %(processName)s \nThread ID - name: %(thread)d - %(threadName)s"
    "\nTime: %(asctime)10s \nDetails: \n\t%(filename)s --> [%(funcName)s():%(lineno)d]\n\t%(levelname)s: %(message)s",
    '%d-%m-%y %H:%M:%S')

debug_Handler = logging.FileHandler(r'.\logs\debug.log', mode='w')
debug_Handler.setLevel(logging.DEBUG)
debug_Handler.setFormatter(output_formatter)

info_Handler = logging.FileHandler(r'.\logs\info.log', mode='w')
info_Handler.setLevel(logging.INFO)
info_Handler.setFormatter(output_formatter)

streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
streamHandler.setFormatter(stream_formatter)

now = int(time.time())
error_mail_handler = logging.handlers.SMTPHandler(mailhost=("smtp.gmail.com", 587),
                                                  fromaddr="archimedes.noreply@gmail.com",
                                                  toaddrs=mailing_list,
                                                  subject='Log report #' + str(now),
                                                  credentials=('archimedes.noreply@gmail.com', 'fXV-r^kZqpZn7yBt'),
                                                  secure=())
error_mail_handler.setFormatter(mail_formatter)
error_mail_handler.setLevel(logging.WARNING)

rootLogger.addHandler(error_mail_handler)
rootLogger.addHandler(debug_Handler)
rootLogger.addHandler(info_Handler)
rootLogger.addHandler(streamHandler)

mpl.rcParams['agg.path.chunksize'] = 5000000
path_to_img = r"D:\Archimedes\Images"

if __name__ == '__main__':
    logging.info('Started')
    start = timeit.default_timer()

    fig1 = plt.figure()
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    fig0 = plt.figure()
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    fig1.suptitle('Data from 22/11/2020')
    fig0.suptitle('Data from 22/11/2020')
    ax1 = fig1.add_subplot()
    ax0 = fig0.add_subplot()
    # af.read_data(day=22, month=11, year=2020, quantity='Error',file_start=1, file_stop=1)
    af.psd(day=22, month=11, year=2020, quantity='Error', ax=ax1, interval=300, mode='low noise')
    # af.time_evolution(day=22, month=11, year=2020, quantity='itf', ax=ax1, show_extra=True)
    # af.time_evolution(day=22, month=11, year=2020, quantity='Pick Off', ax=ax0)
    # mng = plt.get_current_fig_manager
    # mng.window.state('zoomed')
    # fig0.savefig(os.path.join(path_to_img, r'20201122_Data_used.png'))
    # fig1.savefig(os.path.join(path_to_img, r'20201122_ASD.png'))
    # pkl.dump(fig1, open(os.path.join(path_to_img, r'20201122_ASD.pickle'), 'wb'))
    stop = timeit.default_timer()
    print('Time before plt.show(): ', (stop - start), 's')
    print('Time before plt.show(): ', (stop - start) / 60, 'min')
    logging.info('Analysis completed (plt.show excluded)')
    plt.show()
    logging.info('Analysis completed (plt.show included)')
