__author__ = "Luca Pesenti and Davide Rozza "
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.2.0"
__maintainer__ = "Luca Pesenti and Davide Rozza"
__email__ = "l.pesenti6@campus.unimib.it, drozza@uniss.it"
__status__ = "Prototype"

r"""
[LAST UPDATE: 26 Jan 2022 - Luca Pesenti]

This file is to be used as a prototype to launch the functions contained in ET_functions.py
However, it is perfectly functioning

Enjoy the world of seismometers!  :-)
"""

from obspy import UTCDateTime
import ET_functions as ET
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

XML_path = config['Paths']['xml_path']
Data_path = config['Paths']['data_path']
XML_file = config['Paths']['xml_filename']
et_sens = config['Paths']['ET_sens_file']
npz_data = config['Paths']['npz_file']
img_path = config['Paths']['images_dir']

network = config['Instrument']['network']
sensor = config['Instrument']['sensor']
location = config['Instrument']['location']
channel = config['Instrument']['channel']

ti = UTCDateTime(config['Quantities']['start_date'])
Twindow = int(config['Quantities']['psd_window'])
Overlap = float(config['Quantities']['psd_overlap'])
TLong = int(config['Quantities']['TLong'])
means = float(config['Quantities']['number_of_means'])
verbose = config.getboolean('Quantities', 'verbose')
savedata = config.getboolean('Quantities', 'save_data')

Data_path2 = config['Paths']['data_path2']
sensor2 = config['Instrument']['sensor2']

csv_path = config['Paths']['csv_path']
csv_filename = config['Paths']['csv_filename']
csv_filename2 = config['Paths']['csv_filename2']

if __name__ == '__main__':
    # ET.csv_creators(XML_path + XML_file, Data_path, network, sensor, location, channel, ti, Twindow, Overlap, verbose)
    # ET.heatmap_from_csv(multi_csv=True, path_to_csvs=r'D:\ET\2021\Heatmap\csv_files\600', save_img=True)
    # ET.comparison_from_csv(path_to_csv1=csv_path + csv_filename, path_to_csv2=csv_path + csv_filename2)
    # ET.asd_from_csv(csv_path + csv_filename)

    # ET.quantile_plot(XML_path + XML_file, Data_path, network, sensor, location, channel, ti, Twindow, Overlap,
    #                  verbose, save_img=True, xscale='both', show_plot=True, unit='VEL', img_path=img_path)

    # ET.et_sens_single_comparison(et_sens_path=et_sens, npz_file=npz_data, nlnm_comparison=False)
    ET.et_sens_comparison(et_sens_path=et_sens, filexml=XML_path + XML_file, Data_path1=Data_path,
                          Data_path2=Data_path2, network=network, sensor1=sensor, sensor2=sensor2, location=location,
                          channel=channel, tstart=ti, Twindow=Twindow, Overlap=Overlap, TLong=TLong, verbose=verbose,
                          show_plot=True, unit='VEL', save_img=False, nbins=10)
    # Read Inventory and get freq array, response array, sample freq.
    # fxml, respamp, fsxml, gain = ET.read_Inv(XML_path + XML_file, network, sensor, location, channel, ti, Twindow, verbose=verbose)

    # st_tot = ET.extract_stream(XML_path + XML_file, Data_path, network, sensor, location, channel, ti, ti + 1000,
    #                            Twindow,
    #                            verbose=verbose)
    # ET.ppsd(st_tot, XML_path + XML_file, sensor, Twindow, Overlap, temporal=False)
    # ET.spectrogram(XML_path + XML_file, Data_path, network, sensor, location, channel, ti, Twindow, Overlap,
    #                verbose, save_img=False, save_csv=False, xscale='both', show_plot=False)
    # ET.quantile_plot(XML_path + XML_file, Data_path, network, sensor, location, channel, ti, Twindow, Overlap,
    #                  verbose, save_img=True, xscale='both', show_plot=True)
    # ET.rms_comparison(XML_path + XML_file, Data_path, Data_path2, network, sensor, sensor2, location, channel, ti,
    #                   Twindow, verbose, ratio=False, save_img=False, hline=False)

    # st_tot.plot()
    # freq, psd, samp_rate, rms, id = ET.psd_rms_finder(st_tot, XML_path + XML_file, network, sensor, location, channel,
    #                                                   ti, Twindow, Overlap, means, verbose, out=savedata)
    # ET.plot_maker(frequency_data=freq, psd_data=psd, sampling_rate=samp_rate, rms_data=rms, sensor_id=id)
