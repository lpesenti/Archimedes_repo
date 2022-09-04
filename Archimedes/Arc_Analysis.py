__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti"]
__version__ = "1.1.0"
__maintainer__ = "Luca Pesenti (until September 30, 2022)"
__email__ = "lpesenti@uniss.it"
__status__ = "Prototype"

r"""
This file should be used not only as a prototype for the use of the methods contained in the Arc_functions.py, but also
because it contains the creation of two logger, DEBUG and INFO level.
The logs are created in the ./logs folder and each time this script is used they are overwritten.
It is possible to enable loggers by changing the relative booleans in the Arc_config.ini file (for the email logger, 
please note that the mail list and the password of the archimedes mail should be updated).
To change some variable, please refer to the Arc_config.ini before. Inside this file it is possible to change several
variables without modifying the code directly.
"""

import configparser
import logging.handlers
import timeit
from time import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import Arc_functions as af

# ------------------------
# Reading the config file
# ------------------------
config = configparser.ConfigParser()
config.read('Arc_config.ini')

# Email
mailing_list = [x for x in config['Email']['mail_list'].split(',')]

# Paths
path_to_img = config['Paths']['images_dir']

# Booleans
info_logger = config.getboolean('Bool', 'info_logger')
debug_logger = config.getboolean('Bool', 'debug_logger')
email_logger = config.getboolean('Bool', 'email_logger')

# ------------------------
# Creation of the root logger
# ------------------------
rootLogger = logging.getLogger('data_analysis')
rootLogger.propagate = False
rootLogger.setLevel(logging.DEBUG)

# ------------------------
# Formatting the log output
# ------------------------
# Output file
output_formatter = logging.Formatter(
    "[%(asctime)s | %(filename)s %(funcName)15s(), line %(lineno)d] %(levelname)s: %(message)s", '%d-%m-%y %H:%M:%S')

# System output
stream_formatter = logging.Formatter('%(levelname)s: %(message)s')

# Mail output
mail_formatter = logging.Formatter(
    "Process ID - name: %(process)d - %(processName)s \nThread ID - name: %(thread)d - %(threadName)s"
    "\nTime: %(asctime)10s \nDetails: \n\t%(filename)s --> [%(funcName)s():%(lineno)d]\n\t%(levelname)s: %(message)s",
    '%d-%m-%y %H:%M:%S')

# ------------------------
# Creation of the handlers
# ------------------------
# DEBUG
debug_Handler = logging.FileHandler(r'.\logs\debug.log', mode='w')
debug_Handler.setLevel(logging.DEBUG)
debug_Handler.setFormatter(output_formatter)

# INFO
info_Handler = logging.FileHandler(r'.\logs\info.log', mode='w')
info_Handler.setLevel(logging.INFO)
info_Handler.setFormatter(output_formatter)

# SYSTEM
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.INFO)
streamHandler.setFormatter(stream_formatter)

# EMAIL
error_mail_handler = logging.handlers.SMTPHandler(mailhost=("smtp.gmail.com", 587),
                                                  fromaddr="archimedes.noreply@gmail.com",
                                                  toaddrs=mailing_list,
                                                  subject='Log report #' + str(int(time())),
                                                  credentials=(
                                                      'archimedes.noreply@gmail.com', config['Email']['password']),
                                                  secure=())
error_mail_handler.setFormatter(mail_formatter)
error_mail_handler.setLevel(logging.WARNING)

# ------------------------
# Adding handlers to rootLogger
# ------------------------
rootLogger.addHandler(error_mail_handler) if email_logger else ''
rootLogger.addHandler(debug_Handler) if debug_logger else ''
rootLogger.addHandler(info_Handler) if info_logger else ''
rootLogger.addHandler(streamHandler)

mpl.rcParams['agg.path.chunksize'] = 5000000

if __name__ == '__main__':
    logging.info('Started')
    start = timeit.default_timer()

    fig0 = plt.figure(figsize=(19.2, 10.8))
    fig1 = plt.figure(figsize=(19.2, 10.8))
    fig2 = plt.figure(figsize=(19.2, 10.8))
    fig, axs = plt.subplots(nrows=2,
                            ncols=1, sharex=True)
    fig3, axs1 = plt.subplots(nrows=2,
                              ncols=2, sharex=True)
    # fig3, axs1 = plt.subplots(2)
    # fig.suptitle('Temperature control')
    # fig1.suptitle('Loop control')
    # fig0.suptitle('Data from 18/05/2021')
    # fig1.suptitle('Data from 18/05/2021')
    ax0 = fig0.add_subplot()
    ax1 = fig1.add_subplot()
    ax2 = fig2.add_subplot()
    # fig2 = plt.figure()
    # mng = plt.get_current_fig_manager
    # mng.window.state('zoomed')
    # ax2 = fig2.add_subplot()
    # af.psd(day=22, month=11, year=2020, quantity='Error', ax=ax0, ax1=ax1, time_interval=300, mode='low noise', rms_th=7e-12, psd_len=60, low_freq=2, high_freq=20)
    # af.time_evolution(day=2, month=9, year=2022, quantity='itf', show_extra=False, ax=ax0)
    # af.time_evolution(day=22, month=11, year=2020, quantity='itf', file_stop=191, show_extra=False, ax=ax1)
    # af.soe_tevo(day=2, month=9, year=2022, ax=ax0, ndays=1, file_start=1807, file_stop=1807, quantity='temperature', scitype='TEM')
    af.soe_tevo(day=4, month=9, year=2022, ax=axs1[0, 0], ndays=1, quantity='itf', scitype='SCI', file_start=1100)
    af.soe_tevo(day=4, month=9, year=2022, ax=axs1[1, 0], ndays=1, quantity='correction', scitype='SCI',
                file_start=1100)
    # af.soe_tevo(day=3, month=9, year=2022, ax=axs1[0], ndays=1, quantity='correction', scitype='SCI', file_start=1713,
    #             file_stop=1803)
    # af.soe_tevo(day=3, month=9, year=2022, ax=ax1, ndays=1, quantity='actuator 1', scitype='SCI', file_start=1130)
    # af.soe_tevo(day=3, month=9, year=2022, ax=axs1[1], ndays=1, quantity='actuator 2', scitype='SCI', file_start=1130)
    af.soe_tevo(day=4, month=9, year=2022, ax=axs1[0, 1], ndays=1, quantity='temperature', scitype='TEM',
                file_start=1052)
    af.soe_tevo(day=4, month=9, year=2022, ax=axs1[1, 1], ndays=1, quantity='thermal correction', scitype='TEM',
                file_start=1052)
    # af.soe_asd(day=4, month=9, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=1300, file_stop=1400, quantity='correction',
    #            psd_len=1000, pick_off=False, label=r'Error', scitype='SCI')
    # af.soe_asd(day=15, month=6, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=1534, file_stop=1539, quantity='Sum',
    #            psd_len=100, pick_off=False, label=r'$\Sigma$', scitype='OL')
    # af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=116, file_stop=116, quantity='ITF',
    #            psd_len=200, pick_off=False, label='Laser ON - no tubo')
    # af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=823, file_stop=823, quantity='ITF',
    #            psd_len=200, pick_off=False, label='Laser ON - con tubo')
    # af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=901, file_stop=901, quantity='ITF',
    #            psd_len=200, pick_off=False, label='Laser ON - con tubo con schermo campione')
    # af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=613, file_stop=618, quantity='pick off',
    #            psd_len=200, pick_off=False, label='Laser ON - con tubo con specchio con copertura feed')
    # af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=804, file_stop=809, quantity='pick off',
    #            psd_len=200, pick_off=False, label='Laser ON - con tubo con specchio con copertura feed con tubo2')
    # af.easy_psd(day=22, month=11, year=2020, quantity='error', ax=ax0, init_time='23:25:36.094358',
    #             final_time='23:55:35.994314')
    # af.time_evolution(day=18, month=5, year=2021, quantity='itf', ax=ax0)
    # fig0.savefig(os.path.join(path_to_img, fname1))
    # fig1.savefig(os.path.join(path_to_img, fname2))
    # pkl.dump(fig1, open(os.path.join(path_to_img, r'20201122_ASD.pickle'), 'wb'))
    # fig.tight_layout()
    fig0.tight_layout()
    fig.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    # fig0.savefig(os.path.join(path_to_img, 'tEvo.png'), dpi=1200)
    # fig0.savefig(os.path.join(path_to_img, 'tEvo.pdf'), dpi=1200)
    # fig1.savefig(os.path.join(path_to_img, 'tEvo_wclear.png'), dpi=1200)
    # fig1.savefig(os.path.join(path_to_img, 'tEvo_wclear.pdf'), dpi=1200)
    stop = timeit.default_timer()
    print('Time before plt.show(): ', (stop - start), 's')
    print('Time before plt.show(): ', (stop - start) / 60, 'min')
    logging.info('Analysis completed (plt.show excluded)')
    plt.show()
    logging.info('Analysis completed (plt.show included)')
