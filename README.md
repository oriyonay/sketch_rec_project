# Sketch Recognition Project

Repository for our sketch recognition (Texas A&M's CSCE 624) class project, which tracks mouse and eye data to help correct tremors in mouse movements.

## reader.py
This file is used to read the stored data

## tracker.py
This file is used to start the data collection

## tracker_functions.py
This file contains the helper functions and classes to collect mouse and eyetackign coordinates
### class FreshestFrame
This Class is a wrapper class for cv2 video capture that assures there is no race condition between recorded frames so that it is all in order and synced with mouse movements
### class Tracke
This class contains the functions for capturing mouse and eye coordinates.

Group members: Ori Yonay, Rosendo Narvaez, Daniel Bang
