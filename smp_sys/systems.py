"""**smp_sys.systems**

A system in the smp framework is any thing that can be packed into a
box with inputs and outputs and some kind of internal activity that
transforms inputs into outputs.

I distinguish open-loop systems (OLS) and closed-loop systems (CLS).

*OLS*'s are autonomous data sources that ignore any actual input, just
do their thing and produce some output. Examples are a file reader, a
signal generator, ...

*CLS*'s are data sources that depend fundamentally on some input and
will probably not produce any interesting output without any
input. Examples are all actual simulated and real robots, things with
motors, things that can move or somehow do stuff in any kind of world.

"""

# existing systems in legacy code for pulling in
#  - pointmass: mass, dimension, order(0: kinematic/pos,
#    1: dynamic/velocity, 2: dynamic/force),
#   - model v1
#    - smp/smp/ode_inert_system.py: InertParticle, InertParticle2D, InertParticleND (the best)
#    - smp/smpblocks/smpblocks/systems.py
#
#   - model v2
#    - explauto/explauto/environment/pointmass/pointmass.py
#    - smq/smq/robots.py

# smp/smp/arm: kinematic, dynamic
# smp_sphero/sphero: wheels, angle/vel, x/y
# smq/smq/arm
# smq/smq/stdr
# smp/morse_work/atrv
# smp/morse_work/turtlebot
# smp/morse_work/quadrotor
# bha model
# ntrtsim
# malmo

# TODO
#  - A system should be given a reference to a 'world' that implies
#    constraints on state values and provides autonmous activity from outside the agent

from functools import partial
import numpy as np

from smp_base.funcs import *

class SMPSys(object):
    """SMPSys

    Basic smp system class

    :param dict conf: a configuration dictionary

    Takes the config dict and copies all items to corresponding class members
    Checks for presence of ROS libraries
    Initializes pubs/subs dictionaries
    """
    def __init__(self, conf = {}):
        """SMPSys.__init__

        Basic smp system class init

        Arguments:

        - conf: a configuration dictionary

        Takes the config dict and copies all items to corresponding class members
        Checks for presence of ROS libraries
        Initializes pubs/subs dictionaries
        """

        self.conf = conf
        # set_attr_from_dict(self, conf) # check that this is OK

        # set_attr_from_dict_ifs(self, ifs_conf)
        
        # self.id = self.__class__.__name__
        for k, v in conf.items():
            setattr(self, k, v)
            # print "%s.init self.%s = %s" % (self.__class__.__name__, k, v)

        # FIXME: check for sensorimotor delay configuration
        # FIXME: check for motor range configuration
            
        # ROS
        if hasattr(self, 'ros') and self.ros:
            # rospy.init_node(self.name)
            self.subs = {}
            self.pubs = {}
            
    def step(self, x):
        """SMPSys.step

        Basic smp system class step function

        :param numpy.ndarray x: the input column vector

        Does nothing.
        


        :returns: None
        """
        return None
    
################################################################################
# point mass system simple
class PointmassSys(SMPSys):
    """PointmassSys

    A point mass system (pm), which is an abstract model of a rigid
    body robot in an n-dimensional isotropic space. The robot's state
    :math:`x = (x_a, x_v, x_p)^T \in \mathcal{R}^{3 n}` with

    .. math::

        \\begin{eqnarray}
            x_a & := & \\text{acceleration} \\\\
            x_v & := & \\text{velocity} \\\\
            x_p & := & \\text{position} \\\\
        \\end{eqnarray}


    Taken from `smq/robots
    <https://github.com/x75/smq/blob/master/smq/robots.py>`_, and it
    seems to be the same code as in `explauto/environments/pointmass
    <https://github.com/x75/explauto/blob/smp/explauto/environment/pointmass/pointmass.py>`_.

    Missing: noise, motor aberration, transfer funcs, ...
    """
    defaults = {
        'sysdim': 1,
        'x0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'statedim': 3,
        'dt': 1e-1,
        'mass': 1.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.001,
        "sysnoise": 1e-2,
        }
    
    def __init__(self, conf = {}):
        """Pointmass.__init__

        Arguments:
        - conf: configuration dictionary
        -- mass: point _mass_
        -- sysdim: dimension of system, usually 1,2, or 3D
        -- statedim: 1d pointmass has 3 variables (pos, vel, acc) in this model, so sysdim * 3
        -- dt: integration time step
        -- x0: initial state
        -- order: NOT IMPLEMENT (control mode of the system, order = 0 kinematic, 1 velocity, 2 force)
        """
        SMPSys.__init__(self, conf)
        
        # state is (pos, vel, acc).T
        # self.state_dim
        if not hasattr(self, 'x0'):
            self.x0 = np.zeros((self.statedim, 1))
        self.x  = self.x0.copy()
        self.cnt = 0

    def reset(self):
        self.x = self.x0.copy()
        
    def step(self, x = None, world = None):
        """PointmassSys.step

        Update the robot, pointmass
        """
        # print "%s.step[%d] x = %s" % (self.__class__.__name__, self.cnt, x)
        # x = self.inputs['u'][0]
        self.apply_force(x)
        # return dict of state values
        return {'s_proprio': self.compute_sensors_proprio(),
                's_extero':  self.compute_sensors_extero(),
                's_all':     self.compute_sensors(),
        }
        
    def bound_motor(self, m):
        return np.clip(m, self.force_min, self.force_max)

    def apply_force(self, u):
        """control pointmass with force command (2nd order)"""
        # print "u", u, self.mass, u/self.mass
        # FIXME: insert motor transfer function
        a = (u/self.mass).reshape((self.sysdim, 1))
        # a = (u/self.mass).reshape((self.sysdim, 1)) - self.x[:self.sysdim,[0]] * 0.025 # experimental for homeokinesis hack
        # a += np.random.normal(0.05, 0.01, a.shape)

        # world modification
        if np.any(self.x[:self.sysdim] > 0):
            a += np.random.normal(0.05, 0.01, a.shape)
        else:
            a += np.random.normal(-0.1, 0.01, a.shape)
            
        # print("a.shape", a.shape)
        # print "a", a, self.x[self.conf.s_ndims/2:]
        v = self.x[self.sysdim:self.sysdim*2] * (1 - self.friction) + a * self.dt
        
        # self.a_ = a.copy()
        
        
        # # world modification
        # v += np.sin(self.cnt * 0.01) * 0.05
        
        # print "v", v
        p = self.x[:self.sysdim] + v * self.dt

        # collect temporary state description (p,v,a) into joint state vector x
        self.x[:self.sysdim] = p.copy()
        self.x[self.sysdim:self.sysdim*2] = v.copy()
        self.x[self.sysdim*2:] = a.copy()

        # apply noise
        # self.x += self.sysnoise * np.random.randn(self.x.shape[0], self.x.shape[1])
        
        # print "self.x[2,0]", self.x[2,0]

        # self.scale()
        # self.pub()                
        self.cnt += 1
        
        # return x
        # self.x = x # pointmasslib.simulate(self.x, [u], self.dt)

    def compute_sensors_proprio(self):
        return self.x[self.sysdim*2:]
    
    def compute_sensors_extero(self):
        return self.x[self.sysdim:self.sysdim*2]
    
    def compute_sensors(self):
        """compute the proprio and extero sensor values from state"""
        return self.x


class Pointmass2Sys(SMPSys):
    """Pointmass2Sys

    A point mass system (pm), which is an abstract model of a rigid
    body robot in an n-dimensional isotropic space. The robot's state
    :math:`x = (x_a, x_v, x_p)^T \in \mathcal{R}^{3 n}` with

    .. math::

        \\begin{eqnarray}
            x_a & := & \\text{acceleration} \\\\
            x_v & := & \\text{velocity} \\\\
            x_p & := & \\text{position} \\\\
        \\end{eqnarray}


    Fancy model taken from smp/smp/ode_inert_system.py

    Features:
    - Perturbation
    - Non-isotropic forcefield
    - Non-identity motor channel coupling
    - Non-identity motor transfer functions
    """
    defaults = {
        'sysdim': 1,
        'a0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'v0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'x0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'statedim': 3,
        'lag': 1,
        'dt': 1e-1,
        'mass': 1.0,
        'order': 2,
        'forcefield': False,
        'coupling_sigma': 1e-9,
        'transfer': False,
        'force_max':  1.0,
        'force_min': -1.0,
        'friction': 0.001, # 0.012
        'anoise_mean': 0.0,
        'anoise_std': 2e-2,
        'vnoise_mean': 0.0,
        'vnoise_std': 1e-6,
        }
    
    def __init__(self, conf = {}):
        """Pointmass2Sys.__init__

        Arguments:
        - conf: configuration dictionary
        -- mass: point _mass_
        -- sysdim: dimension of system, usually 1,2, or 3D
        -- statedim: 1d pointmass has 3 variables (pos, vel, acc) in this model, so sysdim * 3
        -- dt: integration time step
        -- x0: initial state
        -- order: NOT IMPLEMENT (control mode of the system, order = 0 kinematic, 1 velocity, 2 force)
        """
        SMPSys.__init__(self, conf)
        
        # state is (pos, vel, acc).T
        # self.state_dim
        if not hasattr(self, 'x0'):
            self.x0 = np.zeros((self.statedim, 1))
            
        self.x  = self.x0.copy()
        self.cnt = 0

        bla = dict(a0=0., v0=0., x0=0.,
            numsteps=1000, dt=1, dim=2,
            alag=20, mass = 1.0, order = 2, forcefield = False,
            motortransfer = False, controller = False,
            coupling_sigma = 1e-9,
            transfer = False)
        print "bla", bla
        
        # state vectors
        self.a       = np.zeros((self.sysdim, 1))
        self.a_noise = np.zeros((self.sysdim, 1))
        self.v       = np.zeros((self.sysdim, 1))
        self.v_noise = np.zeros((self.sysdim, 1))
        self.x       = np.zeros((self.sysdim, 1))
        self.u       = np.zeros((self.sysdim, 1))
        # command buffer to simulate motor delay
        self.u_delay = np.zeros((self.sysdim, self.lag))
        print "u_delay", self.u_delay.shape
        
        # reset states
        self.reset()

        ############################################################
        # motor to force coupling: f = h(C * a), with coupling matrix C and transfer function h
        # coupling matrix C, default is Identity
        self.coupling_a_v = np.eye(self.sysdim)
        # non-identity coupling
        self.coupling_a_v_noise(sigma = self.coupling_sigma)

        # coupling transfer functions
        # self.transfer = transfer
        if self.transfer > 0:
            # self.coupling_funcs = [partial(f, a = np.random.uniform(-1, 1), b = np.random.uniform(-1, 1)) for f in [linear, nonlin_1, nonlin_3]] # nonlin_2, 
            self.coupling_funcs = [linear, nonlin_1, nonlin_3]
        else:
            self.coupling_funcs = [identity]
            
        # random;y select a transfer function for each dimension
        self.coupling_func_a_v = np.random.choice(self.coupling_funcs, self.sysdim)
        # debugging coupling transfer functions
        # for f_ in self.coupling_func_a_v:
        #     print "f_", type(f_), f_.__name__, f_(x = 10.0)
        # print "coupling_func_a_v", type(self.coupling_func_a_v)

        
    def reset(self):
        """Pointmass2Sys.reset

        Reset state x to initial state x0
        """
        self.x = self.x0.copy()
        
    def coupling_a_v_noise(self, sigma = 1e-1):
        """Pointmass2Sys.coupling_a_v_noise

        Create additional random coupling between the motor channels and the system dimensions
        """
        print "coupling 0", self.coupling_a_v
        self.coupling_a_v += np.random.uniform(-sigma, sigma, self.coupling_a_v.shape)
        # self.coupling_a_v = np.random.uniform(-sigma, sigma, self.coupling_a_v.shape)
        print "coupling n", self.coupling_a_v

    def coupling_func_a_v_apply(self, x):
        """Pointmass2Sys.coupling_func_a_v_apply

        Element-wise func application
        """
        for i in range(self.sysdim):
            x[i] = self.coupling_func_a_v[i](x[i])
        return x
    
    def step_single(self, u = None):
        """Pointmass2Sys.step

        Compute one integration step
        """
        assert u is not None

        # compute acceleration noise
        self.anoise = np.random.normal(self.anoise_mean, self.anoise_std, size = (1, self.sysdim))

        # motor delay / lag
        self.u_delay[...,[0]] = u.copy()
        u = self.u_delay[...,[-1]].T
        self.u_delay = np.roll(self.u_delay, shift = 1, axis = 1)
        
        # action
        self.a = (u + self.anoise)/self.mass
        # self.a = u/self.mass
        self.a = np.dot(self.coupling_a_v, self.a.T) 
        self.a = self.coupling_func_a_v_apply(self.a)

        # motor out bounding
        self.a = self.bound_motor(self.a)
        
        # vnoise = np.random.normal(self.vnoise_mean, self.vnoise_std,
        #                           size=(1, self.sysdim))
        # self.v_noise[i+1] = vnoise
        
        # 0.99 is damping / friction
        if self.order == 2:
            self.v = self.x[self.sysdim:self.sysdim*2] * (1 - self.friction) + (self.a * self.dt)
            self.p = self.x[:self.sysdim] + self.v * self.dt
        elif self.order == 1:
            self.v = self.a
            # this introduces saturation of position p
            self.p = self.x[:self.sysdim] * (1 - self.friction) + (self.v * self.dt)
            # self.p = self.x[:self.sysdim] + (self.v * self.dt)
        elif self.order == 0:
            self.v = self.a - (self.v * self.friction) # (self.v[i] * (1 - self.friction)) + (self.a[i+1] * self.dt) # + vnoise
            self.p = self.a * 0.2 + self.p * 0.8
            
        # collect temporary state description (p,v,a) into joint state vector x
        self.x[:self.sysdim] = self.p.copy()
        self.x[self.sysdim:self.sysdim*2] = self.v.copy()
        self.x[self.sysdim*2:] = self.a.copy()
                
    # def step2(self, i, dt):
    #     """Pointmass2Sys.step2

    #     Perform n substeps for each step, integratiing at super-resolution
    #     """
    #     # self.step
    #     substeps = 10
    #     dtl = dt/float(substeps)
    #     # print "dtl", dtl
    #     a_ = np.zeros((1, self.sysdim))
    #     a_noise_ = np.zeros((1, self.sysdim))
    #     v = self.v[i].copy()
    #     x = self.x[i].copy()
    #     # print "x pre", x
    #     # print "u", self.u[i-self.lag]
    #     for j in range(substeps):
    #         anoise = np.random.normal(self.anoise_mean, self.anoise_std,
    #                                   size=(1, self.sysdim))
    #         if self.forcefield:
    #             aff = self.force_field(x)
    #         else:
    #             aff = np.zeros_like(x) # 
    #         anoise += aff # add to the other noise
    #         a_noise_ += anoise
    #         # self.a_noise[i+1] = anoise
    #         # with delayed action
    #         a = (self.u[i-self.lag] + anoise)/self.mass
    #         # print "a", a
    #         a_ += a
    #         # print a_
    #         # 0.99 is damping / friction
    #         # self.friction
    #         v = (v * (1 - 0.001)) + (a * dtl) # + vnoise
    #         # self.v[i+1] = (self.v[i] * (1 - self.friction)) + (self.a[i+1] * dt) # + vnoise
    #         # print v[i+1]
    #         x = x.copy() + v * dtl
    #         # print "x inner", x
    #     # print "v", v
    #     self.a[i+1] = a_.copy()
    #     self.a_noise[i+1] = a_noise_
    #     self.v[i+1] = v.copy()
    #     self.x[i+1] = x.copy()
    #     # print "x", x, self.x[i-1], self.x[i], self.x[i+1]
    
    # def step3(self, i, dt):
    #     """Pointmass2Sys.step3

    #     Another step function variation dealing more compactly with temporal indices
    #     """
    #     anoise = np.random.normal(self.anoise_mean, self.anoise_std,
    #                               size=(1, self.sysdim))
    #     # force field
    #     if self.forcefield:
    #         aff = self.force_field(self.x[i])
    #     else:
    #         aff = np.zeros_like(self.x[i]) # 
    #     anoise += aff # add to the other noise
    #     self.a_noise[i] = anoise
        
    #     # FIXME: this is a hack and doesn't expose the final motor signal
    #     # to the obuf
    #     u = self.u[i-self.lag]
    #     if self.motortransfer:
    #         u = self.motortransfer_func(u)
    #         self.u[i-self.lag] = u            
    #     self.a[i] = (u + anoise)/self.mass
        
    #     # 0.99 is damping / friction
    #     # update velocity
    #     self.v[i+1] = (self.v[i] * (1 - self.friction)) + (self.a[i] * dt) # + vnoise
    #     # update position
    #     self.x[i+1] = self.x[i] + self.v[i+1] * dt
        
    # def simulate(self):
    #     for i in range(self.numsteps-1):
    #         # self.control(i)
    #         self.step(i, self.dt)

    # def force_field(self, x):
    #     # pass
    #     return power4(x)
    # return sinc(x)

    # def show_result(self):
    #     import pylab as pl
    #     pl.subplot(311)
    #     pl.plot(self.a * 0.01, label="a")
    #     pl.plot(self.u * 0.1, label="u")
    #     pl.legend()
    #     pl.subplot(312)
    #     pl.plot(self.v, label="v")
    #     pl.legend()
    #     pl.subplot(313)
    #     pl.plot(self.x, label="x")
    #     pl.legend()
    #     pl.show()
        
    def step(self, x = None):
        """Pointmass2Sys.step

        One update step and return system dict
        """
        # print "%s.step[%d] x = %s" % (self.__class__.__name__, self.cnt, x)
        # x = self.inputs['u'][0]
        # self.apply_force(x)
        # self.step_single(self.bound_motor(x))
        self.step_single(x)
        self.cnt += 1
        # return dict of state values
        return {'s_proprio': self.compute_sensors_proprio(),
                's_extero':  self.compute_sensors_extero(),
                's_all':     self.compute_sensors(),
        }
        
    def bound_motor(self, m):
        """Pointmass2Sys.bound_motor

        Bound the motor values to max/min values
        """
        return np.clip(m, self.force_min, self.force_max)

    def apply_force(self, u):
        """control pointmass with force command (2nd order)"""
        # print "u", u, self.mass, u/self.mass
        # FIXME: insert motor transfer function
        a = (u/self.mass).reshape((self.sysdim, 1))
        # a = (u/self.mass).reshape((self.sysdim, 1)) - self.x[:self.sysdim,[0]] * 0.025 # experimental for homeokinesis hack
        # a += np.random.normal(0.05, 0.01, a.shape)

        # world modification
        if np.any(self.x[:self.sysdim] > 0):
            a += np.random.normal(0.05, 0.01, a.shape)
        else:
            a += np.random.normal(-0.1, 0.01, a.shape)
            
        # print("a.shape", a.shape)
        # print "a", a, self.x[self.conf.s_ndims/2:]
        v = self.x[self.sysdim:self.sysdim*2] * (1 - self.friction) + a * self.dt
        
        # self.a_ = a.copy()
        
        
        # # world modification
        # v += np.sin(self.cnt * 0.01) * 0.05
        
        # print "v", v
        p = self.x[:self.sysdim] + v * self.dt

        # collect temporary state description (p,v,a) into joint state vector x
        self.x[:self.sysdim] = p.copy()
        self.x[self.sysdim:self.sysdim*2] = v.copy()
        self.x[self.sysdim*2:] = a.copy()

        # apply noise
        # self.x += self.sysnoise * np.random.randn(self.x.shape[0], self.x.shape[1])
        
        # print "self.x[2,0]", self.x[2,0]

        # self.scale()
        # self.pub()                
        self.cnt += 1
        
        # return x
        # self.x = x # pointmasslib.simulate(self.x, [u], self.dt)

    def compute_sensors_proprio(self):
        return self.x[self.sysdim*2:]
    
    def compute_sensors_extero(self):
        return self.x[self.sysdim:self.sysdim*2]
    
    def compute_sensors(self):
        """compute the proprio and extero sensor values from state"""
        return self.x
    
################################################################################
# simple arm system, from explauto
    
def forward(angles, lengths):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: a tuple (x, y) of the end-effector position

    .. warning:: angles and lengths should be the same size.
    """
    x, y = joint_positions(angles, lengths)
    return x[-1], y[-1]

def joint_positions(angles, lengths, unit='rad'):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: x positions of each joint, y positions of each joints, except the first one wich is fixed at (0, 0)

    .. warning:: angles and lengths should be the same size.
    """
    # print "angles", angles, "lengths", lengths
    
    if len(angles) != len(lengths):
        raise ValueError('angles and lengths must be the same size!')

    if unit == 'rad':
        a = np.array(angles)
    elif unit == 'std':
        a = np.pi * np.array(angles)
    else:
        raise NotImplementedError
     
    a = np.cumsum(a)
    return np.cumsum(np.cos(a)*lengths), np.cumsum(np.sin(a)*lengths)

class SimplearmSys(SMPSys):
    """SimplearmSys

    explauto's simplearm environment

    an n-joint / n-1 segment generative robot
    """

    defaults = {
        'sysdim': 1,
        'x0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'statedim': 3,
        'dt': 1e-1,
        'mass': 1.0,
        "force_max":  1.0,
        "force_min": -1.0,
        "friction": 0.001,
        "sysnoise": 1e-2,
        'dim_s_proprio': 3,
        'length_ratio': [1],
        'm_mins': -1,
        'm_maxs': 1,
        'dim_s_extero': 2,
        }
    
    def __init__(self, conf = {}):
        """SimplearmSys.__init__

        Arguments:
        - conf: configuration dictionary
        -- mass: point _mass_
        -- sysdim: dimension of system, usually 1,2, or 3D
        -- statedim: 1d pointmass has 3 variables (pos, vel, acc) in this model, so sysdim * 3
        -- dt: integration time step
        -- x0: initial state
        -- order: NOT IMPLEMENT (control mode of the system, order = 0 kinematic, 1 velocity, 2 force)
        """
        SMPSys.__init__(self, conf)
        
        # state is (pos, vel, acc).T
        # self.state_dim
        if not hasattr(self, 'x0'):
            self.x0 = np.zeros((self.statedim, 1))
        self.x  = self.x0.copy()
        self.cnt = 0



        self.factor = 1.0

        self.lengths = self.compute_lengths(self.dim_s_proprio, self.length_ratio)

        self.m = np.zeros((self.dim_s_proprio, 1))
        
    def reset(self):
        """SimplearmSys.reset"""
        self.x = self.x0.copy()
        
    # def step(self, x = None, world = None):
    #     """update the robot, pointmass"""
    #     # print "%s.step[%d] x = %s" % (self.__class__.__name__, self.cnt, x)
    #     # x = self.inputs['u'][0]
    #     self.apply_force(x)
    #     # return dict of state values
    #     return {'s_proprio': self.compute_sensors_proprio(),
    #             's_extero':  self.compute_sensors_extero(),
    #             's_all':     self.compute_sensors(),
    #     }
        

    def compute_lengths(self, n_dofs, ratio):
        """SimplearmSys.reset"""
        l = np.ones(n_dofs)
        for i in range(1, n_dofs):
            l[i] = l[i-1] / ratio
        return l / sum(l)

    def compute_motor_command(self, m):
        """SimplearmSys.reset"""
        m *= self.factor
        # print "m", m.shape, "m_mins", self.m_mins, "m_maxs", self.m_maxs
        ret = np.clip(m.T, self.m_mins, self.m_maxs).T
        # print "ret.shape", ret.shape
        return ret

    def compute_sensors_proprio(self):
        """SimplearmSys.reset"""
        # hand_pos += 
        return self.m + self.sysnoise * np.random.randn(*self.m.shape)

    def step(self, x):
        """SimplearmSys.reset

        update the robot, pointmass
        """
        # print "%s.step x = %s" % (self.__class__.__name__, x)
        # print "x", x.shape
        # self.m = self.compute_motor_command(self.m + x)# .reshape((self.dim_s_proprio, 1))
        self.m = self.compute_motor_command(x)# .reshape((self.dim_s_proprio, 1))
        
        # print "m", m
        # self.apply_force(x)
        return {
            # "s_proprio": self.m,
            's_proprio': self.compute_sensors_proprio(),
            "s_extero":  self.compute_sensors_extero(),
            's_all':     self.compute_sensors(),
            }

    def compute_sensors_extero(self):
        """SimplearmSys.reset"""
        # print "m.shape", self.m.shape, "lengths", self.lengths
        hand_pos = np.array(forward(self.m, self.lengths)).reshape((self.dim_s_extero, 1))
        hand_pos += self.sysnoise * np.random.randn(*hand_pos.shape)
        # print "hand_pos", hand_pos.shape
        return hand_pos

    def compute_sensors(self):
        """SimplearmSys.reset"""
        """compute the proprio and extero sensor values from state"""
        # return np.vstack((self.m, self.compute_sensors_extero()))
        return np.vstack((self.compute_sensors_proprio(), self.compute_sensors_extero()))
        # return self.x
    
# class SimpleArmRobot(Robot2):
#     def __init__(self, conf, ifs_conf):
#         Robot2.__init__(self, conf, ifs_conf)
        
#         # self.length_ratio = length_ratio
#         # self.noise = noise

#         self.factor = 1.0

#         self.lengths = self.compute_lengths(self.dim_s_proprio, self.length_ratio)

#         self.m = np.zeros((self.dim_s_proprio, 1))

#     def compute_lengths(self, n_dofs, ratio):
#         l = np.ones(n_dofs)
#         for i in range(1, n_dofs):
#             l[i] = l[i-1] / ratio
#         return l / sum(l)

#     def compute_motor_command(self, m):
#         m *= self.factor
#         return np.clip(m, self.m_mins, self.m_maxs)

#     def step(self, world, x):
#         """update the robot, pointmass"""
#         print "%s.step world = %s, x = %s" % (self.__class__.__name__, world, x)
#         # print "x", x.shape
#         self.m = self.compute_motor_command(self.m + x)# .reshape((self.dim_s_proprio, 1))
        
#         # print "m", m
#         # self.apply_force(x)
#         return {"s_proprio": self.m, # self.compute_sensors_proprio(),
#                 "s_extero": self.compute_sensors_extero()}

#     def compute_sensors_extero(self):
#         hand_pos = np.array(forward(self.m, self.lengths)).reshape((self.dim_s_extero, 1))
#         hand_pos += self.sysnoise * np.random.randn(*hand_pos.shape)
#         # print "hand_pos", hand_pos.shape
#         return hand_pos
                

    
sysclasses = [SimplearmSys, PointmassSys]
# sysclasses = [SimplearmSys]


if __name__ == "__main__":
    """smp_sys.systems.main

    simple test for this file's systems:
    - iterate all known classes
    - run system for 1000 steps on it's own proprioceptive (motor) sensors
    - plot timeseries
    """
    for c in sysclasses:
        print "class", c
        c_ = c(conf = c.defaults)
        c_data = []
        for i in range(1000):
            # do proprio feedback
            x = c_.compute_sensors_proprio() * 0.1
            # print "x", x
            # step system with feedback input
            d = c_.step(x = np.roll(x, shift = 1) * -1.0)['s_all'].copy()
            # print "d", d.shape
            c_data.append(d)

        # print c_data
        # print np.array(c_data).shape

        import matplotlib.pylab as plt
        # remove additional last axis
        plt.plot(np.array(c_data)[...,0])
        plt.show()
