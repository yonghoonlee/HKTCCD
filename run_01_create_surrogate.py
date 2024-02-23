import numpy as np
from smt.surrogate_models import RMTC
import csv
from matplotlib import pyplot as plt
import openmdao.api as om

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

# Create surrogate model
xt = np.array(data)[:,0:2]
yt = np.array(data)[:,2:]
RANGE = np.array([np.min(xt, axis=0), np.max(xt, axis=0)])

sm = RMTC(print_global=False, xlimits=RANGE.transpose())
sm.set_training_values(xt, yt)
sm.train()

# Plot
fg = plt.figure()
N = 31
[X, Y] = np.meshgrid(np.linspace(RANGE[0,0], RANGE[1,0], N), np.linspace(RANGE[0,1], RANGE[1,1], N))
I = np.concatenate((np.reshape(X, (-1,1)), np.reshape(Y, (-1,1))), axis=1)
R = sm.predict_values(I)
plt.contourf(X, Y, np.reshape(R[:,0], (N,N)))
plt.plot(xt[:,0], xt[:,1], 'ko')
plt.show()

prob = om.Problem()
prob.model = om
