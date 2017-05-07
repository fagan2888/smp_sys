"""smp systems

A system in the smp framework is any thing that can be packed into a box with inputs and outputs and some kind of internal
activity that transforms inputs into outputs.

I distinguish open-loop systems (ols) and closed-loop systems (cls).

OLS's are autonomous data sources that ignore any actual input, just do their thing and produce some output. Examples are a file reader,
a signal generator, ...

CLS's are data sources that depend fundamentally on some input and will probably not produce any interesting output without any input. Examples are
all actual simulated and real robots, things with motors, things that can move or somehow do stuff in any kind of world.
"""

# existing system in legacy code
# smp/smp/pointmass: kinematic, dynamic
# smp/smp/arm: kinematic, dynamic
# smp_sphero/sphero: wheels, angle/vel, x/y
# smq/smq/arm
# smq/smq/pm
# smq/smq/stdr
# smp/morse_work/atrv
# smp/morse_work/turtlebot
# smp/morse_work/quadrotor
# ...?

class smpSys(object):
    def __init__(self):
        self.id = self.__class__.__name__

    def step(self, x):
        return 0
