__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti"]
__version__ = "1.0.0"
__maintainer__ = "Luca Pesenti"
__email__ = "l.pesenti6@campus.unimib.it"
__status__ = "Prototype"

import configparser
import logging.handlers
import timeit
from time import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import Arc_functions as af

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

now = int(time())
error_mail_handler = logging.handlers.SMTPHandler(mailhost=("smtp.gmail.com", 587),
                                                  fromaddr="archimedes.noreply@gmail.com",
                                                  toaddrs=mailing_list,
                                                  subject='Log report #' + str(now),
                                                  credentials=(
                                                      'archimedes.noreply@gmail.com', config['DEFAULT']['psw']),
                                                  secure=())
error_mail_handler.setFormatter(mail_formatter)
error_mail_handler.setLevel(logging.WARNING)

# rootLogger.addHandler(error_mail_handler)
rootLogger.addHandler(debug_Handler)
rootLogger.addHandler(info_Handler)
rootLogger.addHandler(streamHandler)

mpl.rcParams['agg.path.chunksize'] = 5000000
path_to_img = config['Paths']['images_dir']

if __name__ == '__main__':
    logging.info('Started')
    start = timeit.default_timer()

    fig0 = plt.figure(figsize=(19.2, 10.8))
    fig1 = plt.figure(figsize=(19.2, 10.8))
    fig2 = plt.figure(figsize=(19.2, 10.8))
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
    # af.time_evolution(day=22, month=11, year=2020, quantity='itf', file_stop=191, show_extra=True, ax=ax0)
    # af.time_evolution(day=22, month=11, year=2020, quantity='itf', file_stop=191, show_extra=False, ax=ax1)
    af.soe_asd(day=5, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=841, file_stop=846, quantity='ITF',
               psd_len=200, pick_off=False, label='Laser OFF')
    af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=116, file_stop=116, quantity='ITF',
               psd_len=200, pick_off=False, label='Laser ON - no tubo')
    af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=823, file_stop=823, quantity='ITF',
               psd_len=200, pick_off=False, label='Laser ON - con tubo')
    af.soe_asd(day=6, month=5, year=2022, ax=ax0, ax1=ax1, ax2=ax2, file_start=901, file_stop=901, quantity='ITF',
               psd_len=200, pick_off=False, label='Laser ON - con tubo con schermo campione')
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
    fig0.tight_layout()
    fig1.tight_layout()
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
