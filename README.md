# Sketch Recognition Project
Group members: Ori Yonay, Rosendo Narvaez, Daniel Bang

Repository for our sketch recognition (Texas A&M's CSCE 624) class project, which tracks mouse and eye data to help correct tremors in mouse movements.

## reader.py
This file is used to read the stored data

## Main.py
This file is used to start the data collection

## Tracker.py
This file contains the classes and functions to capture mouse and eye movement
### class FreshestFrame
This Class is a wrapper class for cv2 video capture that assures there is no race condition between recorded frames so that it is all in order and synced with mouse movements
### class Tracker
This class contains the functions for capturing mouse, eye, and pupil coordinates.
