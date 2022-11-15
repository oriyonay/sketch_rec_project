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

# parse all args! (vars function converts from Namespace to a dictionary)
args = parser.parse_args()

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

    print('recording stopped. saving data...')

    # save the data:
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)

    print('data saved successfully. thank you!')
