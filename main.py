'''
tracker.py: a mouse (and later, mouse & eye) tracker

mouse movement recording is done using the pynput library
eye tracking is done using TBD
'''

# native imports
import argparse
import pickle
import time
# library imports
import numpy as np
from pynput.mouse import Controller
#Import functions that we made
from tracker import *
from tqdm import tqdm
# ------------------------- PARSE CLI ARGUMENTS ------------------------- #
# create the argument parser
parser = argparse.ArgumentParser()

# add arguments:
# sample rate (number of samples to be recorded per second)
parser.add_argument('--sr', type=int, default=500, required=False)

# number of seconds to record
parser.add_argument('--record_length', type=int, default=2, required=False)

# where to save the pickled file?
parser.add_argument('--out', type=str, default='./osu_2_person_1_1080p_vid_output.p', required=False)

# which derivatives of data to compute / save? (will compute UP TO this derivative,
# i.e., a value of 2 means we compute first and second derivative; 0 means no
# derivatives will be computed)
parser.add_argument('--derivatives', type=int, default=0, required=False)

# parse all args! (vars function converts from Namespace to a dictionary)
args = parser.parse_args()

# ------------------------- UTILITY FUNCTIONS ------------------------- #
# utility function to compute the derivative of a numpy array.
# x: the array to compute the derivative for
# degree: (optional): the degree of derivative to calculate
def derivative(x, degree=1):
    for _ in range(degree):
        x = np.gradient(x)
    return x

# utility function similar to the one above, but for simultaneously computing
# x and y derivatives for a list of (x, y) tuples:
def derivative_xy(coords, degree=1):
    # convert to x, y numpy arrays:
    x = np.array([x for x, _ in coords])
    y = np.array([y for _, y in coords])

    # compute derivatives:
    dx = derivative(x, degree)
    dy = derivative(y, degree)

    # convert back to (x, y) coordinates:
    coords = [(xi, yi) for xi, yi in zip(dx, dy)]
    return coords

# ------------------------- RECORD DATA ------------------------- #
if __name__ == '__main__':
    print('initializing...')
    tracker = Tracker(0)
    print('intialized!')

    # calculate the sampling interval (1 / sr) and total
    # number of samples to be recorded:
    interval = 1.0 / args.sr
    n_samples = args.record_length * args.sr

    # create the x, y lists for mouse and eye tracking data:
    data = {'mouse_data' : [], 'eye_data_right' : [], 'eye_data_left' : [], 'time' : []}

    print('beginning recording!')

    # record the mouse & eye positions n_samples times:
    for _ in tqdm(range(n_samples)):

        # Collecting Eye coordinates and mouse coordinates
        p,e,t = tracker.capture()

        e_r = None
        e_l = None

        if e is not None:

            e_r = e[0]
            e_l = e[1]

        # append this data to our data:
        data['mouse_data'].append(p)
        data['eye_data_right'].append(e_r)
        data['eye_data_left'].append(e_l)
        data['time'].append(t)

        # sleep for interval amount of time (until next sampling):
        # time.sleep(interval)

    tracker.stop_camera()
    print('recording stopped.')

    # compute derivatives if requested:
    if args.derivatives > 0:
        print('computing derivatives...')
        for degree in range(1, args.derivatives+1):
            # compute nth degree derivatives:
            d_mouse = derivative_xy(data['mouse_data'], degree=degree)
            # d_eye = derivative_xy([data['eye_data']], degree=degree)

            # save into data dictionary:
            data[f'mouse_data_d{degree}'] = d_mouse
            # data[f'eye_data_d{degree}'] = d_eye

    # save the data:
    print('saving data...')
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)

    print('data saved successfully. thank you!')
