"""**smp_sys.systems**

.. moduleauthor:: Oswald Berthold, 2017

Depends: numpy, smp_base

A system in the smp framework is any thing that can be packed into a
box with inputs and outputs and some kind of internal activity that
transforms inputs into outputs. We distinguish open-loop systems (OLS)
and closed-loop systems (CLS).

**OLS**'s are autonomous data sources that ignore any actual input, just
do their thing and autonomously produce some output. Examples are file
readers, signal generators, and other read-only information
taps. **CLS**'s are data sources that depend fundamentally on some input
and will probably not produce any interesting output without any
input. Examples are all actual simulated and real robots, things with
motors, things that can move or somehow do stuff in any kind of real
or simulated world.

Classically systems are clear cut and well established concepts. In
the self-exploration context, systems are seen as instances of
explanation challenges or problems. Ideally we would like to be able
to sample systems from the space of all problems conditioned on some
essential parameters or system properties which control the degrees of
difficulty of the explanation problems. This implies systematic
knowledge of problem difficulty and systematic knowledge of adequate
agents for a given level of difficulty.

Current systems in here are all closed-loop ones and include
:class:`SMPSys`, :class:`PointmassSys`, :class:`Pointmass2Sys`,
:class:`smp_sys.BhaSimulatedSys`, :class:`smp_sys.STDRCircularSys`,
:class:`smp_sys.LPZBarrelSys`, :class:`smp_sys.SpheroSys`.

TODO:
 - fix order 0 to be zero, pm and others, intrinsic order for 'real' systems
 - fix mappings from intrinsic modalities and dimensions to configured dims
 - add spatial constraints / environments to these basic systems to be
   able to show indirect inference of external space from prediction violations
   caused by constraints
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

# abstract style
# smp/smp/arm: kinematic, dynamic
# smp_sphero/sphero: wheels, angle/vel, x/y
# smq/smq/arm
# bha model
# pendel
# cartpole
# doppelwippe
# doppelpendel

# real style
# smq/smq/stdr
# smp/morse_work/atrv
# smp/morse_work/turtlebot
# smp/morse_work/quadrotor
# real copter
# festo linear actuator
# ntrtsim
# malmo

# TODO
#  - A system should be given a reference to a 'world' that implies
#    constraints on state values and provides autonmous activity from
#    outside the agent

import copy
from functools import partial
import numpy as np
from pprint import pformat

# import transfer functions: linear, nonlin_1, nonlin_2, ...
from smp_base.funcs import *

# model funcs from smp_graphs, FIXME: move that into smp_base
# from smp_graphs.funcs_models import model
from smp_base.models_funcmapped import model

import logging
from smp_base.common import get_module_logger
logger = get_module_logger(modulename = 'systems', loglevel = logging.INFO)

# dummy block ref
class bla(object):
    pass

class SMPSys(object):
    """SMPSys class

    Basic smp system class
     - Takes the configuration dict and copies all items to
       corresponding class members
     - Checks for presence of ROS libraries
     - Initializes pubs/subs dictionaries

    Defaults:
     - order(int): system order is the order of its difference equation (depth of the recurrence relation?)
     - dims(dict): the system's variables as name:config pairs
     - mem:(float): the amount of intrinsic memory

    Variables in `dims` are configured with:
     - dim(int): dimension of the vector variable
     - dist(float): distance of variable to proprioception (experimental: information distance?)
     - initial(spec): initial state, either an array or a distribution with parameters
     - stats(dict): dict of statistical moments like mean and variance

    Arguments:
     - conf(dict): configuration dictionary
    """
    defaults = {
        'order': 0,
        'dims': {
            's0': {'dim': 1, 'dist': 0.},
        },
        'mem': 1,
        'cnt': 0,
    }
    def __init__(self, conf = {}):
        """SMPSys.__init__
        """
        # prepare the conf
        self.conf = {}
        # set base class and self defaults
        self.conf.update(SMPSys.defaults, **self.defaults)
        # update with instance conf
        self.conf.update(conf)
        
        # copy self.conf to self attributes
        self.__dict__.update(copy.deepcopy(self.conf))
        
        # for k, v in self.conf.items():
        #     setattr(self, k, v)
        #     # print "%s.init self.%s = %s" % (self.__class__.__name__, k, v)

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
    """Pointmass system class

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

    Arguments:
     - conf: configuration dictionary

    Configuration:
     - mass: mass parameter [1.0]
     - sysdim: dimension of system, usually 1,2, or 3D
     - statedim: 1d pointmass has 3 variables (pos, vel, acc) in this model, so sysdim * 3
     - dt: integration time step
     - x0: initial state
     - order: NOT IMPLEMENT (control mode of the system, order = 0 kinematic, 1 velocity, 2 force)

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
        """
        SMPSys.__init__(self, conf)
        
        # state is (pos, vel, acc).T
        # self.state_dim
        # if not hasattr(self, 'x0'):
        #     self.x0 = np.zeros((self.statedim, 1))
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
        'order': 0,
        'dims': {
            # 'm0': {'dim': 1, 'dist': 0, 'initial': np.zeros((1, 1)), 'lag': 1},
            's0': {'dim': 1, 'dist': 0, 'initial': np.random.uniform(-1.0, 1.0, (1, 1))} # mins, maxs
        },
        'x': {}, # initialize x
        'lag': 1,
        'dt': 1e-1,
        'mass': 1.0,
        # 
        'a0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'v0': np.random.uniform(-0.3, 0.3, (3, 1)),
        'x0': np.random.uniform(-0.3, 0.3, (3, 1)),
        # 'statedim': 3,
        'forcefield': False,
        'coupling_sigma': 1e-9,
        'transfer': False, # integer index, 0 means identity
        'force_max':  1.0,
        'force_min': -1.0,
        'friction': 0.001, # 0.012
        'anoise_mean': 0.0,
        'anoise_std': 2e-2,
        'vnoise_mean': 0.0,
        'vnoise_std': 1e-6,
        'numelem': 101,
    }
    
    def __init__(self, conf = {}):
        """Pointmass2Sys.__init__

        Arguments:
        - conf: configuration dictionary
        -- mass: point _mass_
        -- sysdim: dimension of system, usually 1,2, or 3D
        -- dt: integration time step
        -- x0: initial state
        -- order: not implemented (control mode of the system, order = 0 kinematic, 1 velocity, 2 force)
        # -- statedim: 1d pointmass has 3 variables (pos, vel, acc) in this model, so sysdim * 3
        """
        # init parent, set defaults and update with config
        SMPSys.__init__(self, conf)

        # make sure variable description matches with order
        self.check_dims_order()
        
        # make sure variable description has at least one motor entry
        self.check_dims_motor()
        
        # reset states
        self.reset()

        ############################################################
        # motor to force coupling: f = h(C * a), with coupling matrix C and transfer function h
        # FIXME: better coupling is a tensor with shape C x lookup-length, or matrix of functions
        # coupling matrix C, default is Identity
        self.coupling_a_v = np.eye(self.sysdim)
        
        # add noise to coupling matrix
        self.coupling_a_v_noise(sigma = self.coupling_sigma)

        # coupling transfer functions
        self.transfer = int(self.transfer)
        self.coupling_funcs_all = [
            [identity],
            [partial(f, a = np.random.uniform(-1, 1), b = np.random.uniform(-1, 1)) for f in [linear, nonlin_1, nonlin_3]], # nonlin_2,
            [linear, nonlin_1, nonlin_2, nonlin_3],
            self.get_transfer_lookup(),
            [partial(nonlin_cosine, a = 1.0)],
            [partial(nonlin_cosine, a = np.pi/2, b = -np.pi/2)],
            [np.tanh],
        ]
        
        # if self.transfer > 0:
        #     self.coupling_funcs_1 = 
        #     self.coupling_funcs_2 = 

        #     # self.coupling_funcs = [linear, nonlin_1, nonlin_2, nonlin_3] + self.transfer_lookup
        #     # logger.debug('transfer_lookup = %s', self.transfer_lookup)
        #     self.coupling_funcs = self.transfer_lookup + self.coupling_funcs_1
        # else:
        #     self.coupling_funcs = [identity]
            
        # randomly select a transfer function for each dimension
        assert self.transfer < len(self.coupling_funcs_all), "systems.Pointmass2Sys transfer = %s >= len(transfer_list) = %s" % (self.transfer, len(self.coupling_funcs_all))
        self.coupling_func_a_v = np.random.choice(self.coupling_funcs_all[self.transfer], self.sysdim)
        logger.debug("    transfer function choice = %s" % (self.coupling_func_a_v, ))
        
        # debugging coupling transfer functions
        # for f_ in self.coupling_func_a_v:
        #     print "f_", type(f_), f_.__name__, f_(x = 10.0)
        # print "coupling_func_a_v", type(self.coupling_func_a_v)

        # special case order 0
        if self.order == 0:
            self.x['s1'] = self.x['s0']

    def get_transfer_lookup(self):
        self.ref = bla()
        setattr(self.ref, 'cnt', 0)
        setattr(self.ref, 'inputs', {'x': {'val': np.zeros((self.sysdim, 1))}})
        setattr(self.ref, 'y', np.zeros((self.sysdim, 1)))
        # self.mref = bla()
        # setattr(self.mref, 'x', self.ref.inputs['x']['val'].copy())
        # setattr(self.mref, 'y', self.ref.y.copy())
        # self.ref.inputs['x']['val'][i] = x[i]
        self.transfer_model = []
        self.transfer_lookup = []
        # loop over system dimensions / motor dimensions
        for sysdim_ in range(self.sysdim):
            # FIXME: model.get_random_conf()
            d_a = np.random.uniform(0.8, 0.98)
            l_a = np.random.uniform(0.0, 1 - d_a)
            s_a = np.random.uniform(0.0, 1 - d_a - l_a)
            e_a = np.random.uniform(0.0, 1 - d_a - l_a - s_a)
            mconf = {
                'type': 'random_lookup',
                'numelem': self.numelem, # sampling grid
                'l_a': l_a, # 0.0,
                'd_a': d_a, # 0.98,
                'd_s': 0.3,
                's_a': s_a,
                's_f': 2.0,
                'e': e_a,
            }
            # logger.debug("    mconf = %s" % (mconf, ))
            # FIXME: model.get_block_random_conf()
            conf = {
                'params': {
                    'inputs': {
                        'x': {'shape': (self.sysdim, 1)},
                        },
                    },
                }
                        
            self.transfer_model.append(model(ref = self.ref, conf = conf, mref = 'random_lookup', mconf = mconf))
            self.transfer_lookup.append(
                partial(
                    self.transfer_model[-1].predict2,
                    self.ref,
                    self.transfer_model[-1]
                    # ref = self.ref,
                    # mref = self.transfer_model
                )
            )

            return self.transfer_lookup
            
    def check_dims_order(self):
        """check_dims

        Make sure dims spec is complete
         - :attr:`dims` requires an sN entry for every N in [0, ..., order]
        """
        # scan order indices
        for o in range(self.order + 1):
            ordk = 's%d' % (o, )
            # add dim if necessary
            if ordk not in self.dims:
                self.dims[ordk] = {'dim': self.sysdim, 'dist': float(o), 'initial': np.random.uniform(-1, 1, (self.sysdim, 1))}
                logger.debug("adding variable %s = %s to comply with system order %d", ordk, self.dims[ordk], self.order)

    def check_dims_motor(self):
        mks = [k for k in list(self.dims.keys()) if k.startswith('m')]
        # print "motor keys = %s" % (mks, )
        # no motor definitions
        if len(mks) < 1:
            # add default motor at order 0
            mks.append('m0')
            self.dims['m0'] = {'dim': self.sysdim, 'dist': 0, 'initial': np.zeros((self.sysdim, 1)), 'lag': self.lag}
        # sort motor keys
        mks.sort()
        # infer highest order of motor input
        self.order_motor = int(mks[0][1:])
            
    def reset(self):
        """Pointmass2Sys.reset

        Reset state x to initial state x0
        """
        # state vector: array or dict? set initial state from config
        for dk, dv in list(self.dims.items()):
            logger.debug("state dimension key dk = %s, val dv = %s", dk, dv)
            # required entries
            if 'initial' not in dv:
                dv['initial'] = np.random.uniform(-1, 1, (dv['dim'], 1))
            if 'dissipation' not in dv:
                dv['dissipation'] = 0.0

            # motor special: augment dims with additional motor variables
            if dk[0] == 'm': # dk.startswith('m'):
                # fix missing lag
                if 'lag' not in dv:
                    # logger.debug("        dimv is missing lag param, setting dimv.lag to global lag = %d" % (self.lag, ))
                    dv['lag'] = self.lag
                # add motor delayline entry
                self.x[self.get_k_plus(dk, 'd')] = np.zeros((dv['dim'], dv['lag'] + 1))
                # add predicted motor entry (the input)
                self.x[self.get_k_plus(dk, 'p')] = np.zeros((dv['dim'], 1))
                # logger.debug("        dimv.lag = %d" % (dv['lag'], ))
            # else:
            
            # init from conf
            self.x[dk] = dv['initial']
                
        # self.x = self.x0.copy()

    def get_k_plus(self, k = 'm0', plus = 'd'):
        """get augmented dims key from key and modification index 'plus'
        """
        if plus is 'd':
            return '%s_delayline' % (k, )
        elif plus is 'p':
            return '%s_pre' % (k, )
                
    def coupling_a_v_noise(self, sigma = 1e-1):
        """Pointmass2Sys.coupling_a_v_noise

        Create additional random coupling between the motor channels and the system dimensions
        """
        # print "coupling 0", self.coupling_a_v
        self.coupling_a_v += np.random.uniform(-sigma, sigma, self.coupling_a_v.shape)
        # self.coupling_a_v = np.random.uniform(-sigma, sigma, self.coupling_a_v.shape)
        # print "coupling n", self.coupling_a_v

    def coupling_func_a_v_apply(self, x):
        """Pointmass2Sys.coupling_func_a_v_apply

        Element-wise func application
        """
        # logger.debug('coupling_func_a_v_apply x = %s', x)
        for i in range(self.sysdim):
            # legacy transfer funcs
            x[i] = self.coupling_func_a_v[i](x[i])

            # # lookup transfer from funcs_models
            # self.ref.inputs['x']['val'][i] = x[i].copy()
            # # logger.debug('coupling_func_a_v_apply x[%d] = %s, ref.inputs[\'x\'] = %s, ref.y[%d] = %s, mref.y[%d] = %s', i, x[i], self.ref.inputs['x'], i, self.ref.y[i], i, self.mref.y[i])
            # # self.coupling_func_a_v[i](ref = self.ref)
            # self.coupling_func_a_v[i]()
            # # logger.debug('coupling_func_a_v_apply x[%d] = %s, mref.y[%d] = %s', i, x[i], i, self.mref.y[i])
            # # x[i] = self.mref.y[i].copy()
            
        # logger.debug('coupling_func_a_v_apply x = %s, mref.y = %s', x, self.transfer_model.y[i])
        return x
    
    def step_single(self, u = None):
        """Pointmass2Sys.step_single

        Compute one integration step of state update equation.

        .. note:: order indices are reversed w.r.t. math. notation, so that order :math:`o_{i = 0}`
                  always refers to the system's highest of integration, and :math:`o_{i = |o|}`
                  refers to its lowest order of integration
        """
        assert u is not None
        if not type(u) is dict:
            u = {'m%d' % (self.order_motor, ): u.copy()}

        # compute acceleration noise
        self.anoise = np.random.normal(self.anoise_mean, self.anoise_std, size = (1, self.sysdim))

        # apply motor delay / lag
        # iterate over motor groups (vectors) of motor variables
        #  - transforming a predicted value into a measured value
        #  - by applying a channel-specific (vector component) delay
        #  - by applying a transfer function (lookup table, distortion d, smoothness s)
        #  - by applying constraints (self limits)
        #  - by adding entropy
        for mk, mv in [(k, v) for k, v in list(self.dims.items()) if k[0] == 'm']:
            # get delayed motor prediction key
            dlmk = self.get_k_plus(mk, 'd')
            # get instantaneous motor measurement key
            premk = self.get_k_plus(mk, 'p')
            # print "dlmk", dlmk, self.x[dlmk], "u", u

            # store current prediction and feed it into delay line
            self.x[premk] = u[mk].copy()
            self.x[dlmk][...,[0]] = u[mk].copy()
            # self.x[mk] = self.x[dlmk][...,[-1]].T

            # get delayed prediction data
            # logger.debug("    step_single motor loop motorkey = %s, x[motorkey_delay] = %s, lag = %s", mk, self.x[dlmk], mv['lag'])
            a = self.x[dlmk][...,[-1]].T.copy()
            # logger.debug("    step_single motor loop motorkey = %s,             raw a = %s", mk, a)

            # update delay line by one step
            shift_ = 1
            self.x[dlmk] = np.roll(self.x[dlmk], shift = shift_, axis = 1)
            # logger.debug("    step_single motor loop motorkey = %s, x[motorkey_delay] = %s, shift = %s", mk, self.x[dlmk], shift_)
                        
            # apply intrinsice motor noise and divide by mass to get net force
            a = (a + self.anoise)/self.mass
            # logger.debug("    step_single motor loop motorkey = %s, norm a = %s", mk, a)

            # apply coupling transformation
            a = np.dot(self.coupling_a_v, a.T)
            # logger.debug("    step_single motor loop motorkey = %s,  dot a = %s", mk, a)
            
            # apply component-wise transfer func
            a = self.coupling_func_a_v_apply(a)
            # logger.debug("    step_single motor loop motorkey = %s, tran a = %s", mk, a)

            # apply self limits / motor out bounding
            a = self.bound_motor(a)
            # logger.debug("    step_single motor loop motorkey = %s, boun a = %s", mk, a)

            # update internal state
            # logger.debug("    step_single motor loop motorkey = %s, motorval = %s, motorlag = %s", mk, self.x[mk], mv['lag'])
            self.x[mk] = a.copy()
        
        # vnoise = np.random.normal(self.vnoise_mean, self.vnoise_std,
        #                           size=(1, self.sysdim))
        # self.v_noise[i+1] = vnoise

        # integrate system: loop over system order
        for o in range(self.order + 1):
            ordk = 's%d' % (o,)
            # ordk_ = 's%d' % (min(self.order, o+1),)
            ordk_ = 's%d' % (max(0, o - 1),)
            # print "ordk", ordk, ordk_

            # 20180129 reassume
            # motor input at every order
            # motor input from own past (memory / qtap)
            
            # this assumes motor input always at highest order
            self.x[ordk] *= 1 - self.dims[ordk]['dissipation']
            x_tm1 = self.x[ordk]
            # u_t = 0.
            dx_t = (self.x[ordk_].copy() * self.dt)
            if ordk == ordk_: # at bottom order index / top order
                x_tm1 = 0
                
            if 'm%d' % o in self.dims: # have explicit input at order
                mk_ = 'm%d' % (max(0, o - 1), )
                orddlmk = mk_
                # orddlmk = self.get_k_plus(mk_, 'd') # 'm%d' % (max(0, o - 1), )
                u_t = self.x[orddlmk] # * self.dt
                # logger.debug("        step_single order loop motorkey = %s, motorval = %s, dx_t = %s, u_t = %s", orddlmk, self.x[orddlmk], dx_t, u_t)
                dx_t += u_t
                
                # logger.debug("        step_single order loop motorkey = %s, dx_t = %s", orddlmk, dx_t)
                
            # integrating
            self.x[ordk] = x_tm1 + dx_t
            # logger.debug("    step_single order loop orderkey = %s, sensorval = %s = %s + %s", ordk, self.x[ordk], x_tm1, dx_t)
            # print "self.x[ordk]", ordk, self.x[ordk]
            
        # # 0.99 is damping / friction
        # if self.order == 2:
        #     self.v = self.x[self.sysdim:self.sysdim*2] * (1 - self.friction) + (self.a * self.dt)
        #     self.p = self.x[:self.sysdim] + self.v * self.dt
        # elif self.order == 1:
        #     self.v = self.a
        #     # this introduces saturation of position p
        #     self.p = self.x[:self.sysdim] * (1 - self.friction) + (self.v * self.dt)
        #     # self.p = self.x[:self.sysdim] + (self.v * self.dt)
        # elif self.order == 0:
        #     self.v = self.a - (self.v * self.friction) # (self.v[i] * (1 - self.friction)) + (self.a[i+1] * self.dt) # + vnoise
        #     self.p = self.a * 0.2 + self.p * 0.8
            
        # # collect temporary state description (p,v,a) into joint state vector x
        # self.x[:self.sysdim] = self.p.copy()
        # self.x[self.sysdim:self.sysdim*2] = self.v.copy()
        # self.x[self.sysdim*2:] = self.a.copy()

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
        # logger.debug("    step[%d]                  x = %s", self.cnt, x)
        # print "%s.step[%d] x = %s" % (self.__class__.__name__, self.cnt, x)
        # x = self.inputs['u'][0]
        # self.step_single(self.bound_motor(x))
        self.step_single(x)
        self.cnt += 1
        # return dict of state values
        # return {'s_proprio': self.compute_sensors_proprio(),
        #         's_extero':  self.compute_sensors_extero(),
        #         's_all':     self.compute_sensors(),
        # }
        rdict = dict([(dk, self.compute_sensors(dk)) for dk in list(self.dims.keys())])
        # legacy hack
        rdict['s_all'] = self.x['s0']
        if self.order == 0:
            rdict['s1'] = self.x['s0']
        # logger.debug("    step        rdict['s0'] = %s", rdict['s0'])
        return rdict

    def bound_motor(self, m):
        """Pointmass2Sys.bound_motor

        Bound the motor values to max/min values
        """
        return np.clip(m, self.force_min, self.force_max)

    def compute_sensors_proprio(self):
        """Pointmass2Sys.compute_sensors_proprio"""
        # return self.x[self.sysdim*2:]
        return self.x['s0']
    
    def compute_sensors_extero(self):
        """Pointmass2Sys.compute_sensors_extero"""
        if self.order > 0: # and self.x.has_key('s1'):
            return self.x['s1'] # self.x[self.sysdim:self.sysdim*2]
        else:
            return np.zeros_like(self.x['s0'])
    
    def compute_sensors(self, k = None):
        """Pointmass2Sys.compute_sensors

        Compute the proprio and extero sensor values from ground truth state
        """
        if k is None:
            return np.vstack((self.compute_sensors_proprio(), self.compute_sensors_extero())) # self.x

        if not k.endswith('delayline'):
            return self.x[k]
    
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
        'lag': 1,
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
        # check defaults
        for k in list(self.defaults.keys()):
            if k not in conf:
                conf[k] = self.defaults[k]
            assert k in conf, "conf doesn't have required attribute %s, defaults = %s" % (k, self.defaults[k])
        
        SMPSys.__init__(self, conf)
        
        # state is (pos, vel, acc).T
        # self.state_dim
        if not hasattr(self, 'x0'):
            self.x0 = np.zeros((self.statedim, 1))
        self.x  = self.x0.copy()
        self.cnt = 0

        # command buffer to simulate motor delay
        self.u_delay = np.zeros((self.dim_s_proprio, self.lag + 1))

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
        
        self.u_delay[...,[0]] = self.m.copy()
        self.m = self.u_delay[...,[-1]]
        self.u_delay = np.roll(self.u_delay, shift = 1, axis = 1)
        
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
                

    
sysclasses = [SimplearmSys, PointmassSys, Pointmass2Sys]

if __name__ == "__main__":
    """smp_sys.systems.main

    simple test for this file's systems:
    - iterate all known classes
    - run system for 1000 steps on it's own proprioceptive (motor) sensors
    - plot timeseries
    """
    import matplotlib.pylab as plt

    plt.ion()
    fig = plt.figure(figsize=(len(sysclasses) * 5, 3))
    plt.show()
    
    for c_i, c in enumerate(sysclasses):
        print("class", c)
        c_ = c(conf = c.defaults)
        c_data = []

        # minimal stepping
        for i in range(1000):
            # do proprio feedback
            x = c_.compute_sensors_proprio() * 0.1
            # print "x", x
            # step system with feedback input
            d = c_.step(x = np.roll(x, shift = 1) * -1.0)['s_all'].copy()
            # print "d", d.shape
            c_data.append(d)

        # print(c_data)
        # print np.array(c_data).shape
        print(pformat(list(c_.x)))
        print(pformat(list(d)))

        ax = fig.add_subplot(1, len(sysclasses), c_i + 1)
        # remove additional last axis
        ax.set_title(c.__name__)
        ax.plot(np.array(c_data)[...,0])
        plt.draw()

    plt.ioff()
    plt.show()
