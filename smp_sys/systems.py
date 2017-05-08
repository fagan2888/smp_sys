"""smp systems

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
#    - smp/smp/ode_inert_system.py: InertParticle, InertParticle2D, InertParticleND,
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
# ...?

import numpy as np

class SMPSys(object):
    def __init__(self, conf = {}):
        # self.id = self.__class__.__name__
        for k, v in conf.items():
            setattr(self, k, v)
            # print "%s.init self.%s = %s" % (self.__class__.__name__, k, v)

    def step(self, x):
        return None

class PointmassSys(SMPSys):
    """point mass system (pm)

a pm is an abstract model of a rigid body robot represented by the coordinates 

taken from smq/robots.py, seems to be the same code as in
explauto/environments/pointmass.py
"""
    def __init__(self, conf = {}):
        """Pointmass.__init__

params:
 - mass: point _mass_
 - sysdim: dimension of system, usually 1,2, or 3D
 - order: control mode of the system, order = 0 kinematic, 1 velocity, 2 force
"""
        SMPSys.__init__(self, conf)
        
        # state is (pos, vel, acc).T
        # self.state_dim
        self.x0 = np.zeros((self.statedim, 1))
        self.x  = self.x0.copy()
        self.cnt = 0

    def reset(self):
        self.x = self.x0.copy()
        
    def step(self, x = None, world = None):
        """update the robot, pointmass"""
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
        # print "u", u
        # FIXME: insert motor transfer function
        a = (u/self.mass).reshape((self.sysdim, 1))
        # a += np.random.normal(0.05, 0.01, a.shape)

        # # world modification
        # if np.any(self.x[:self.sysdim] > 0):
        #     a += np.random.normal(0.05, 0.01, a.shape)
        # else:
        #     a += np.random.normal(-0.1, 0.01, a.shape)
            
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
        self.x += self.sysnoise * np.random.randn(self.x.shape[0], self.x.shape[1])

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
