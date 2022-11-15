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

# ------------------------- PARSE CLI ARGUMENTS ------------------------- #
# create the argument parser
parser = argparse.ArgumentParser()

# add arguments:
# sample rate (number of samples to be recorded per second)
parser.add_argument('--sr', type=int, default=100, required=True)

# number of seconds to record
parser.add_argument('--record_length', type=int, default=10, required=True)

# where to save the pickled file?
parser.add_argument('--out', type=str, default='./tracker_output.p', required=True)

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

    # create the mouse controller object
    mouse = Controller()

    # calculate the sampling interval (1 / sr) and total
    # number of samples to be recorded:
    interval = 1.0 / args.sr
    n_samples = args.record_length * args.sr

    # create the x, y lists for mouse and eye tracking data:
    data = {'mouse_data' : [], 'eye_data' : []}

    print('beginning recording!')

    # record the mouse & eye positions n_samples times:
    for _ in range(n_samples):
        # get mouse position:
        p = mouse.position

        # get eye position: TODO
        e = None

        # append this data to our data:
        data['mouse_data'].append(p)
        data['eye_data'].append(e)

        # sleep for interval amount of time (until next sampling):
        time.sleep(interval)

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
