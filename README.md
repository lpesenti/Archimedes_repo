# Archimedes_repo

Welcome to the GitHub repository of the Archimedes Experiment in which are present also scripts for the ET experiment!

## Archimedes

###### This directory contains every scripts used to analyze the data acquired during the characterization of the experiment.

###### Moreover, the directory logs contains all the logs file produced by the functions used.

#### ./logs

* Inside this folder are saved the logs that are automatically produced by several functions in _Arc_functions.py_
* Note that the logs are saved only if the relative option is enabled in the _Arc_config.ini_ file

#### ./Results

* In this directory are contained three files (_August 25, 2022_):
    * Arc_article.txt -> Data from
      the [Picoradiant tiltmeter and direct ground tilt measurements at the Sos Enattos site](https://link.springer.com/article/10.1140/epjp/s13360-021-01993-w)
      article.
    * Arc_data.txt -> Results obtained _probably_ by using the 'low interval' mode of the psd() method inside 
      _Arc_functions.py_. However, the loop correction was not used since it is missing.
    * Arc_VirgoData_Jul2019.txt -> Data taken with the prototype while at the Virgo site (_July 2021_)

#### Analysis_Archi_lf_fpga.py (_No longer supported or maintained_)

* An old script used to perform PSD/ASD analysis on Archimedes data.

#### Arc_Analysis.py

* This file has to be considered as an example of how _Arc_functions.py_ should be used.
* This script requires _Arc_functions.py_ in order to work.
* Be aware to change the value of the ___path_to_img___ to the directory where you want to save your images.

#### Arc_common.py

* This script contains all the common and useful functions that are not strictly related to the data analysis.

#### Arc_compressor.py (_No longer supported or maintained_)

* This script was used to perform some compression test on data taken by the Archimedes prototype. However, it is in an
  unstable and unreviewed version.

#### Arc_config.ini

* In this file are contained several variables needed for the analysis on the Archimedes experiment data.
* The values of the variable present must be changed especially the ones related to paths variables.

#### Arc_functions.py

* ___A stable version has been release on April 27, 2021___
* Several variables can be changed in the relative _Arc_config.ini_ file contained in the same directory.
* In this file are stored different useful functions that can be implemented in other script (_see
  Arc_Data_Analysis.py_). Each of them are fully commented, and they should be self-explaining. However, at the
  beginning of the scripts there is a short introduction to the data file format and to other useful quantities.
* Some functions (indicated in their description as 'OLD') were made to work with an older version of
  the filename. Some bugs should be expected when using them with newer data. However, they should work greatly.
* After May 2022, the data have the filename SCI_yy-mm-dd_HHMM.lvm or OL_yy-mm-dd_HHMM.lvm. The former refers to data
  taken by the Archimede experiment, while the latter to the data of the optical lever system. __NOTE__ that the 'OLD'
  functions have never been tested on the optical lever data.
* Please note that at the start of the script there are different variables that can be modified in according to the
  analysis you want to perform.
* Change ___path_to_data___ in the _Arc_config.ini_ to the correct path to your data folder. The data until May 2022
  __MUST__ be saved in the following format!
    * ___path_to_data___
        * _SosEnattos_Data_yyyymmdd_ (1)
            * *.lvm
        * _SosEnattos_Data_yyyymmdd_ (2)
            * *.lvm
        * ...
        * _SosEnattos_Data_yyyymmdd_ (n)
            * *.lvm
* In the latest updates (_April 26, 2022_), two functions were added. Both of them start with _soe..._  and can be used
  to
  easily make the plot of the data taken on the Sos Enattos site. The data __DON'T__ need to be saved in a particular
  folder since it is specified in the _Arc_config.ini_. In fact, the ___path_to_data___ variable refers to the directory
  in which are contained the data as:
    * ___path_to_data___
        * _SCI/OL_yy-mm-dd_HHMM.lvm_ (1)
        * _SCI/OL_yy-mm-dd_HHMM.lvm_ (2)
        * ...
        * _SCI/OL_yy-mm-dd_HHMM.lvm_ (n)
* Note that the _soe..._ functions can distinguish SCI files from OL files through the 'scitype' attribute.

#### Arc_grapher.py

* Simple script needed to reproduce the data publish in
  the [Picoradiant tiltmeter and direct ground tilt measurements at the Sos Enattos site](https://link.springer.com/article/10.1140/epjp/s13360-021-01993-w)
  article.
* Manually can be uncommented the plot relative to the main sources of noise of the Archimedes prototype.

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

#### ET_RMS_TimeEvolution.py

* This script is used to evaluate the integral under the asd evaluated by the _ET_Quantile.py_ and then plot the time
  evolution of this quantity.
* This is a completely ___self-contained___ script which should not be modified.
    * All the variable that must be modified are contained in the relative config file (see _RMS_time_config.ini_)

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
        * _skip_quant_eval_ (Bool): If True the script does not evaluate the quantile curves (use it only if the code
          has
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

#### RMS_time_config.ini

* In this file are contained several variables needed for the _ET_RMS_TimeEvolution.py_ file.
* Below is a description of the variables contained in the file:
    * [Paths]
        * _outDF_path_ (str): The path to the directory in which the script will store the DataFrame created
    * [Quantities]
        * _psd_window_ (int): Length of the PSD expressed in seconds
        * _integral_min_ (float): The lower limit of the integral
        * _integral_max_ (float): The upper limit of the integral

#### config.ini

* In this file are contained several variables needed for the seismic analysis.
* The values of the variable present must be changed especially the ones related to paths variables.

#### Analysis.py

* This module contains different functions used for the analysis of the seismometer.
* See _GUI_Analysis.py_ in which this module is used.

#### GUI_Analysis.py

* This is the Gui used fo the analysis of the sensors placed in the site.
* Please note that this script requires _Analysis.py_ in order to work.





