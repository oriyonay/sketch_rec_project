# Sketch Recognition Project
Group members: Ori Yonay, Rosendo Narvaez, Daniel Bang

Repository for our sketch recognition (Texas A&M's CSCE 624) class project, which tracks mouse and eye data to help correct tremors in mouse movements.

to run use the following command:
python main.py --sr [sample rate] --record_length [seconds of recording] --out [file path for pickle output] --derivatives [integer for nth derivative]

## Important folders
### data
This folder contains all of the data collected in the form of pickle and csv files. It also contains the jupyter notebook used for inference.

### visualizations
Contains visualizations of the data that we used in our presentations and paper


## Python Files
### reader.py
This file is used to read the stored data
(use this as an example on how to access data)
### Main.py
This file is used to start the data collection

### Converter.py
Contains scripts to convert our pickle files into CSV files for inference

### Tracker.py
This file contains the classes and functions to capture mouse and eye movement

#### class FreshestFrame
This Class is a wrapper class for cv2 video capture that assures there is no race condition between recorded frames so that it is all in order and synced with mouse movements
#### class Tracker
This class contains the functions for capturing mouse, eye, and pupil coordinates.
