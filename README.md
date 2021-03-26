# Archimedes-Sassari

Welcome to the GitHub repository of the Archimedes Experiment!

## UNISCO

###### This folder contains different script used during the UNISCO project. Here you find some scripts that are not correlated to the Archimedes experiment.

#### Poisson_plotter.py

* This script plot an animation of the comparison between Poisson distribution and the Normal distribution.
* The animation is saved in _.gif_ format.

#### Thomas.py

* The script evaluates the G factor and its error for a three plane telescope.
* It requires __configThomas.txt__ to work (_see the file for further information_).
* It produces a _.txt_ output, called __Thomas_output.txt__ where are stored every useful information

## Archimedes

###### This directory contains every scripts used to analyze the data acquired during the characterization of the experiment.

#### Arc_functions.py
* In this file are stored different useful functions that can be implemented in other script (_for example see Arc_Data_Analysis.py_).
* At the moment (_26/03/2021_) the functions contained are relative to the time evolution analysis of the data, and the 
  relative method to read the files with this information. Every function is fully commented.
* Please note that at the start of the script there are different variables that can be modified in according to the analysis you want to perform.
* Change ___path_to_data___ value, to the correct path to your data folder. The data __MUST__ be saved in the following format!
  * Data_folder
    * _SosEnattos_Data_yyyymmdd_ (1)
      * *.lvm
    * _SosEnattos_Data_yyyymmdd_ (2)
      * *.lvm
    * ...
    * _SosEnattos_Data_yyyymmdd_ (n)
      * *.lvm
#### Arc_Data_Analysis.py
* This script requires _Arc_functions.py_ in order to work.
* Be aware to change the value of the ___path_to_img___ to the directory where you want to save your images.



