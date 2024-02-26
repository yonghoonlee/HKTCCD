import numpy as np
from smt.surrogate_models import RMTC
import csv
import openmdao.api as om
from hkt_system import hkt_generator

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

sm = RMTC(print_global=False, xlimits=RANGE.transpose(), extrapolate=True)
sm.set_training_values(xt, yt)
sm.train()

# Plot power and omega
N = 51
V_min = 0.18
V_max = V_min/np.min(xt[:,0])**2
PLOT = False

if PLOT:
    from matplotlib import pyplot as plt
    [X, Y] = np.meshgrid(np.linspace(RANGE[0,0], RANGE[1,0], N), np.linspace(RANGE[0,1], RANGE[1,1], N))
    I = np.concatenate((np.reshape(X, (-1,1)), np.reshape(Y, (-1,1))), axis=1)
    R = sm.predict_values(I)
    fg0, ax0 = plt.subplots(nrows=2, ncols=2)
    fg0.set_figwidth(12.0)
    fg0.set_figheight(9.0)
    # plot 1
    ctr0 = ax0[0,0].contourf(X, Y, np.reshape(R[:,0], (N,N)), levels=10)
    pdt0 = ax0[0,0].plot(xt[:,0], xt[:,1], 'ko')
    ax0[0,0].set_xlabel('Duct contraction ratio [-]')
    ax0[0,0].set_ylabel('Blade pitch angle [deg]')
    ax0[0,0].set_title('Power [mW]')
    fg0.colorbar(ctr0, ax=ax0[0,0])
    # plot 2
    ctr1 = ax0[0,1].contourf(X, Y, np.reshape(R[:,1], (N,N))*(2.0*np.pi/60.0), levels=10)
    #pdt1 = ax0[0,1].plot(xt[:,0], xt[:,1], 'ko')
    ax0[0,1].set_xlabel('Duct contraction ratio [-]')
    ax0[0,1].set_ylabel('Blade pitch angle [deg]')
    ax0[0,1].set_title('Angular velocity [rad/s]')
    fg0.colorbar(ctr1, ax=ax0[0,1])
    # plot 3
    ctr2 = ax0[1,0].contourf(V_min/(X*X), Y, np.reshape(R[:,0], (N,N)), levels=10)
    #pdt2 = ax0[1,0].plot(V_min/(xt[:,0]*xt[:,0]), xt[:,1], 'ko')
    ax0[1,0].set_xlabel('Rotor area water velocity [m/s]')
    ax0[1,0].set_ylabel('Blade pitch angle [deg]')
    ax0[1,0].set_title('Power [mW]')
    fg0.colorbar(ctr2, ax=ax0[1,0])
    # plot 4
    ctr3 = ax0[1,1].contourf(V_min/(X*X), Y, np.reshape(R[:,1], (N,N))*(2.0*np.pi/60.0), levels=10)
    #pdt3 = ax0[1,1].plot(V_min/(xt[:,0]*xt[:,0]), xt[:,1], 'ko')
    ax0[1,1].set_xlabel('Rotor area water velocity [m/s]')
    ax0[1,1].set_ylabel('Blade pitch angle [deg]')
    ax0[1,1].set_title('Angular velocity [rad/s]')
    fg0.colorbar(ctr3, ax=ax0[1,1])
    fg0.tight_layout()
    fg0.savefig('fig_power_omega.pdf', format='pdf')

# Problem definition
nn = 61    # nn=61 for 1 minutes with dt=1.0s
dt = 1.0    # dt=1.0s
prob = om.Problem()
prob.model = om.Group()
prob.model.add_subsystem(name='hkt_generator', subsys=hkt_generator(num_nodes=nn, dt=dt, sm=sm, V_min=V_min, V_max=V_max), promotes=['*'])
prob.model.add_subsystem(name='hkt_lackofdata', subsys=om.ExecComp(
        'data_feasibility = 2.0*effective_duct_contraction_ratio + 0.25*pitch - 3.25',
        data_feasibility = {'shape': (nn,), 'units': None},
        effective_duct_contraction_ratio = {'shape': (nn,), 'units': None},
        pitch = {'shape': (nn,), 'units': 'deg'},
        do_coloring=False,
    ),
    promotes=['*'])

prob.model.add_design_var(name='duct_contraction_ratio', lower=0.5, upper=1.0)
prob.model.add_design_var(name='pitch', lower=-12.0, upper=10.0)
prob.model.add_constraint('effective_duct_contraction_ratio', lower=0.5, upper=1.0)
prob.model.add_constraint('pitch', lower=-12.0, upper=10.0)
prob.model.add_constraint('data_feasibility', lower=None, upper=0.0)
prob.model.add_objective('avg_power', ref=-1)

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-3
# prob.driver.options['disp'] = True
# prob.driver.opt_settings['eps'] = 1e-4

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = 'IPOPT'
prob.driver.options['print_results'] = True
prob.driver.opt_settings['print_level'] = 5


prob.setup()
t = np.linspace(0.0, float(nn-1)*dt, nn)
V_water = (2.0*V_min + V_min*np.sin(t))
prob.set_val('t', val=t)
prob.set_val('V_water', val=V_water)
prob.set_val('pitch', np.zeros(t.shape))
prob.set_val('duct_contraction_ratio', np.ones(t.shape))
#prob.run_model()
prob.run_driver()

power = prob.get_val('power')
omega = prob.get_val('omega')
V_throat = prob.get_val('V_throat')

print(prob)