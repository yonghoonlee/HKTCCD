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
N = 51
fg, ax = plt.subplots(nrows=1, ncols=2)
fg.set_figwidth(10.0)
fg.set_figheight(4.5)
[X, Y] = np.meshgrid(np.linspace(RANGE[0,0], RANGE[1,0], N), np.linspace(RANGE[0,1], RANGE[1,1], N))
I = np.concatenate((np.reshape(X, (-1,1)), np.reshape(Y, (-1,1))), axis=1)
R = sm.predict_values(I)
ctr0 = ax[0].contourf(X, Y, np.reshape(R[:,0], (N,N)), levels=10)
#pdt0 = ax[0].plot(xt[:,0], xt[:,1], 'ko')
ax[0].set_xlabel('Rotor-area water velocity [m/s]')
ax[0].set_ylabel('Blade pitch angle} [deg]')
fg.colorbar(ctr0, ax=ax[0])
ctr1 = ax[1].contourf(X, Y, np.reshape(R[:,1], (N,N)), levels=10)
#pdt1 = ax[1].plot(xt[:,0], xt[:,1], 'ko')
ax[1].set_xlabel('Rotor-area water velocity [m/s]')
ax[1].set_ylabel('Blade pitch angle} [deg]')
fg.colorbar(ctr1, ax=ax[1])
fg.show()

prob = om.Problem()
prob.model = om
