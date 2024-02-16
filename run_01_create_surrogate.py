import numpy as np
from smt.surrogate_models import KRG
import csv
import pickle as pkl

# Reading CSV file
data = []
with open('datax.csv', 'rt') as fid:
    csvreader = csv.reader(fid, delimiter=',')
    for row in csvreader:
        try:
            new_data = [float(n) for n in row]
            data.append(new_data)
        except:
            pass

# Scaling
data = np.array(data)
lb = np.tile(np.min(data, axis=0), (data.shape[0], 1))
ub = np.tile(np.max(data, axis=0), (data.shape[0], 1))
data_scaled = (data - lb)/(ub - lb)
lb_x = lb[0, 0:2]
ub_x = ub[0, 0:2]
lb_y = lb[0, 2:]
ub_y = ub[0, 2:]

# Create KRG surrogate model
xt = data_scaled[:, 0:2]
yt = data_scaled[:, 2:]

sm = KRG()
sm.set_training_values(xt, yt)
sm.train()

# Testing SM
# x = np.array([[0.55104, -11]])
# x_scaled = (x - lb_x)/(ub_x - lb_x)
# y_scaled = sm.predict_values(x_scaled)
# y = lb_y + (ub_y - lb_y)*y_scaled
# print(x)
# print(y)

# Saving SM
with open('datax.pkl', 'wb') as fid:
    pkl.dump(sm, fid)

