# Archimedes-Sassari

Welcome to the GitHub repository of the Archimedes Experiment!

## Archimedes

###### This directory contains every scripts used to analyze the data acquired during the characterization of the experiment.

###### Moreover, the directory logs contains all the logs file produced by the functions used.

#### Arc_functions.py

* ___A stable version has been release in 27/04/2021___
* In this file are stored different useful functions that can be implemented in other script (_for example see
  Arc_Data_Analysis.py_).
* At the moment (_27/04/2021_) this package contains several functions. In particular are contained two main functions
  _psd()_ and  _time_evolution()_ which are used for the preliminary studies of the experiment.
* Please note that at the start of the script there are different variables that can be modified in according to the
  analysis you want to perform.
* Change ___path_to_data___ value, to the correct path to your data folder. The data __MUST__ be saved in the following
  format!
    * Data_folder
        * _SosEnattos_Data_yyyymmdd_ (1)
            * *.lvm
        * _SosEnattos_Data_yyyymmdd_ (2)
            * *.lvm
        * ...
        * _SosEnattos_Data_yyyymmdd_ (n)
            * *.lvm
* In the latest updates (_26/04/2022_), two functions were added. Both of them start with _soe..._  and can be used to
  easily make the plot of the data taken on the Sos Enattos site. The data __DON'T__ need to be saved in a particular
  folder since it is specified in the config.ini

#### Arc_common.py

* This script contains all the common and useful functions that are not strictly related to the data analysis.

#### Arc_Data_Analysis.py

* This file has to be considered as an example of how _Arc_functions.py_ should be used.
* This script requires _Arc_functions.py_ in order to work.
* Be aware to change the value of the ___path_to_img___ to the directory where you want to save your images.

#### Arc_grapher.py

* Simple script needed to reproduce the data publish in
  the [Picoradiant tiltmeter and direct ground tilt measurements at the Sos Enattos site](https://link.springer.com/article/10.1140/epjp/s13360-021-01993-w)
  article.
* Manually can be uncommented the plot relative to the main sources of noise of the Archimedes prototype.

#### Arc_compressor.py

* This script was used to perform some compression test on data taken by the Archimedes prototype. However, it is in an
  unstable and unreviewed version.

#### config.ini

* In this file are contained several variables needed for the analysis on the Archimedes experiment data.
* The values of the variable present must be changed especially the ones related to paths variables.

## ET_sensors

###### This folder contains different script used for the characterization of the Sardinia site of ET.

#### ET_Analysis.py

* This script has to be used as a prototype to launch the functions contained in _ET_functions.py_. However, it is
  perfectly working.
* This script needs the _config.ini_ file contained it this folder which has all the variables settings.

#### ET_functions.py

* In this are stored several functions needed to perform seismic analysis. All the functions contained are fully
  commented except where indicated.
* This file is currently under developing. Some changes must be expected.

#### ET_common.py

* This script contains all the common and useful functions that are not strictly related to the data analysis.

#### ET_Quantile.py

* This script is used to evaluate the quantile curve of a given seismometer dataset.
* This is a completely ___self-contained___ script which should not be modified.
    * All the variable that must be modified are contained in the relative config file (see _Quantile_config.ini_)

#### Quantile_config.ini

* In this file are contained several variables needed for the _ET_Quantile.py_ file.
* Below is a description of the variables contained in the file:
    * [DEFAULT]
        * _verbose_ (Bool):  If True enable more verbosity
        * _only_daily_ (Bool): If True the script simply create daily dataframe without doing anything else
        * _skip_daily_ (Bool): If True the script does not create the daily dataframe (use it only if the code has
          already run before)
        * _skip_freq_df_ (Bool): If True the script does not create the frequency dataframe (use it only if the code has
          already run before)
        * _skip_quant_eval_ (Bool): If True the script does not evaluate the quantile curves (use it only if the code has
          already run before)
        * _unit_ (str): It could be 'ACC' or 'VEL' (up to version 0.1 only the 'ACC' option is supported)
    * [Paths]
        * _xml_path_ (str): It is the path to the .xml file used to read the inventory in which is contained the
          response amplitude and other information
        * _data_path_ (str): The data path on which to perform the analysis
        * _outDF_path_ (str): The path to the directory in which the script will store the DataFrame created
    * [Instrument]
        * _network_ (str): The network of the seismometer considered on which perform the analysis
        * _sensor_ (str): The name of the sensor on which perform the analysis
        * _location_ (str): The location of the sensor on which perform the analysis
        * _channel_ (str): The channel of the sensor on which perform the analysis
    * [Quantities]
        * _psd_window_ (int): Length of the PSD expressed in seconds
        * _TLong_ (int): The time windows in which the data will be divided to speed up PSD evaluation expressed in
          seconds (this quantity must be greater or equal to the _psd_window_ parameter)
        * _psd_overlap_ (float): The overlap used in the PSD evaluation (from 0 to 1)
        * _quantiles_ (list): The list of quantiles you wish to calculate

#### config.ini

* In this file are contained several variables needed for the seismic analysis.
* The values of the variable present must be changed especially the ones related to paths variables.

#### Analysis.py

* This module contains different functions used for the analysis of the seismometer.
* See _GUI_Analysis.py_ in which this module is used.

#### GUI_Analysis.py

* This is the Gui used fo the analysis of the sensors placed in the site.
* Please note that this script requires _Analysis.py_ in order to work.





