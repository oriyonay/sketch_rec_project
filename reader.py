import pickle

file_name = "tracker_output.p"
with open(file_name, 'rb') as f:
    x = pickle.load(f)

p = x["mouse_data"]
e = x["eye_data"]
t = x["time"]
print(p)
print(e)
print(t)

print(len(p))
print(len(e))
print(len(t))