import pickle
import matplotlib.pyplot as plt

file_name = "osu1.p"
with open(file_name, 'rb') as f:
    x = pickle.load(f)

p = x["mouse_data"]
e_r = x["eye_data_right"]
e_l = x["eye_data_left"]
t = x["time"]
# print(p)
# print(e)
# print(t)
#
# print(len(p))
# print(len(e))
# print(len(t))

def plotter(points,x_label = "x",y_label = "y"):
    x = []
    y = []
    for point in points:
        if point!=None:
            print(point)
            x.append(point[0])
            y.append(point[1])
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

plotter(p)
plotter(e_r)
plotter(e_l)