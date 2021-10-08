import obspy
from obspy import UTCDateTime
import ET_functions as ET

XML_path = r'/Users/drozza/Downloads/'
Data_path = r'/Users/drozza/Downloads/P3-P2-preliminary/BKP_Sardegna_P2/2021/09/'
XML_file = 'cjunkk_ETScope_1.xml'
network = 'ET'
sensor = 'P2'
location = '01'
channel = 'HHZ'
ti = UTCDateTime("2021-09-14T00:00:00")
Twindow = 300
verbose = True
Overlap = 0

if __name__ == '__main__':
    # Read Inventory and get freq array, response array, sample freq.
    # fxml, respamp, fsxml, gain = ET.read_Inv(XML_path + XML_file, network, sensor, location, channel, ti, Twindow, verbose=verbose)
    st_tot = ET.extract_stream(XML_file, Data_path, network, sensor, location, channel, ti, ti + 1000, Twindow,
                             verbose=verbose)
    #ET.ppsd(st_tot, XML_path + XML_file, sensor, Twindow, Overlap)
    ET.psd_rms_finder(st_tot, XML_path + XML_file, network, sensor, location, channel, ti, Twindow, Overlap)
