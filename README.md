# Archimedes-Sassari

Welcome to the GitHub repository of the Archimedes Experiment!

## Archimedes
###### This directory contains every scripts used to analyze the data acquired during the characterization of the experiment.

#### Arc_functions.py

* ___A stable version has been release in 27/04/2021___
* In this file are stored different useful functions that can be implemented in other script (_for example see 
  Arc_Data_Analysis.py_).
* At the moment (_27/04/2021_) this package contains several functions. In particular are contained two main functions 
  _psd()_ and  _time_evolution()_ which are used for the preliminary studies of the experiment.
* Please note that at the start of the script there are different variables that can be modified in according to the
  analysis you want to perform.
* Change ___path_to_data___ value, to the correct path to your data folder. The data __MUST__ be saved in the 
  following format!
  * Data_folder
    * _SosEnattos_Data_yyyymmdd_ (1)
      * *.lvm
    * _SosEnattos_Data_yyyymmdd_ (2)
      * *.lvm
    * ...
    * _SosEnattos_Data_yyyymmdd_ (n)
      * *.lvm
  
#### Arc_common.py

* Thi script contains all the common and useful functions that are not strictly related to the data analysis.

#### Arc_Data_Analysis.py

* This file has to be considered as an example of how _Arc_functions.py_ should be used.
* This script requires _Arc_functions.py_ in order to work.
* Be aware to change the value of the ___path_to_img___ to the directory where you want to save your images.

## ET_sensors
###### This folder contains different script used for the characterization of the Sardinia site of ET.

#### Analysis.py

* This module contains different functions used for the analysis of the seismometer.
* See _GUI_Analysis.py_ in which this module is used.

#### GUI_Analysis.py

* This is the Gui used fo the analysis of the sensors placed in the site.
* Please note that this script requires _Analysis.py_ in order to work.





