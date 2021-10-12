from obspy import UTCDateTime
import ET_functions as ET
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

XML_path = config['Paths']['xml_path']
Data_path = config['Paths']['data_path']
XML_file = config['Paths']['xml_filename']

network = config['Instrument']['network']
sensor = config['Instrument']['sensor']
location = config['Instrument']['location']
channel = config['Instrument']['channel']

ti = UTCDateTime(config['Quantities']['start_date'])
Twindow = float(config['Quantities']['psd_window'])
Overlap = float(config['Quantities']['psd_overlap'])
means = float(config['Quantities']['number_of_means'])
verbose = config['Quantities']['verbose']


if __name__ == '__main__':
    # Read Inventory and get freq array, response array, sample freq.
    # fxml, respamp, fsxml, gain = ET.read_Inv(XML_path + XML_file, network, sensor, location, channel, ti, Twindow, verbose=verbose)
    st_tot = ET.extract_stream(XML_path + XML_file, Data_path, network, sensor, location, channel, ti, ti + 1000, Twindow,
                             verbose=verbose)
    ET.ppsd(st_tot, XML_path + XML_file, sensor, Twindow, Overlap)
    ET.psd_rms_finder(st_tot, XML_path + XML_file, network, sensor, location, channel, ti, Twindow, Overlap, means, verbose)
