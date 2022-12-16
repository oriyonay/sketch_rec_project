import pickle
import matplotlib.pyplot as plt
import numpy as np

def csv_converter(data,filename):

    header = ["time","mouse_x","mouse_y","right_eye_x","right_eye_y","left_eye_x","left_eye_y"]
    rows = []
    rows.append(header)

    for i in range(len(data["time"])):
        row = []
        t = data["time"][i]
        p_x = data["mouse_data"][i][0]
        p_y = data["mouse_data"][i][1]
        e_r_x = data["eye_data_right"][i][0]
        e_r_y = data["eye_data_right"][i][1]
        e_l_x = data["eye_data_left"][i][0]
        e_l_y = data["eye_data_left"][i][1]

        row.append(t)
        row.append(p_x)
        row.append(p_y)
        row.append(e_r_x)
        row.append(e_r_y)
        row.append(e_l_x)
        row.append(e_l_y)

        rows.append(row)
    np.savetxt(filename,rows,delimiter=", ",fmt='% s')


file_name = "./data/osu1.p"
with open(file_name, 'rb') as f:
    x = pickle.load(f)


csv_converter(x,"osu1.csv")