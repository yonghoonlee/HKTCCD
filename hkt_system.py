import numpy as np
import openmdao.api as om

class hkt_generator(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('num_nodes', types=int)
        self.options.declare('dt', types=float)
        self.options.declare('sm')
        self.options.declare('V_min')
        self.options.declare('V_max')
    
    def setup(self):

        nn = self.options['num_nodes']
        dt = self.options['dt']
        sm = self.options['sm']
        V_min = self.options['V_min']
        V_max = self.options['V_max']
        t = np.linspace(0.0, float(nn-1)*dt, nn)

        # given inputs
        self.add_input('t', shape=(nn,), val=t, units='s')
        self.add_input('V_water', shape=(nn,), units='m/s')

        # control variables
        self.add_input('pitch', shape=(nn,), units='rad')
        self.add_input('duct_contraction_ratio', shape=(nn,), units=None)

        # outputs
        self.add_output('V_throat', shape=(nn,), units='m/s')
        self.add_output('power', shape=(nn,), units='W')
        self.add_output('omega', shape=(nn,), units='rad/s')

    def compute(self, inputs, outputs):

        nn = self.options['num_nodes']
        dt = self.options['dt']
        sm = self.options['sm']
        V_min = self.options['V_min']
        V_max = self.options['V_max']

        # inputs
        t = inputs['t']
        V_water = inputs['V_water']
        pitch = inputs['pitch']
        duct_contraction_ratio = inputs['duct_contraction_ratio']
        V_throat = V_water / duct_contraction_ratio**2
        effective_duct_contraction_ratio = np.sqrt(V_min / V_throat)

        R = sm.predict_values(np.concatenate((effective_duct_contraction_ratio.reshape((-1, 1)), pitch.reshape((-1, 1))), axis=1))
        power = R[:,0].flatten()
        omega = R[:,1].flatten()

        # outputs
        outputs['V_throat'] = V_throat
        outputs['omega'] = omega
        outputs['power'] = power

