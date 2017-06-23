"""smp_sys

ROS based systems: simulators and real robots

stdr, lpzrobots, MORSE
sphero, car, turtlebot, quad, puppy, nao
"""

import time
import numpy as np
from collections import OrderedDict

try:
    import rospy
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
except Exception, e:
    print "Import rospy failed with %s" % (e, )


from smp_sys.systems import SMPSys


class STDRCircularSys(SMPSys):

    defaults = {
        'ros': True,
        'dt': 0.1,
        'dim_s_proprio': 2, # linear, angular
        'dim_s_extero': 3,
        'outdict': {},
        'smdict': {},
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
            self.x0 = np.zeros((self.dim_s_proprio, 1))
        self.x  = self.x0.copy()
        self.cnt = 0

        if not self.ros:
            import sys
            print "ROS not configured but this robot (%s) requires ROS, exiting" % (self.__class__.__name__)
            sys.exit(1)
            
        # timing
        self.rate = rospy.Rate(1/self.dt)
            
        # pub / sub
        from nav_msgs.msg      import Odometry
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg   import Range

        # actuator / motors
        self.twist = Twist()
        self.twist.linear.x = 0
        self.twist.linear.y = 0
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0
        self.twist.angular.z = 0

        # odometry
        self.odom = Odometry()
        self.sonars = []
        self.outdict = {
            "s_proprio": np.zeros((self.dim_s_proprio, 1)),
            "s_extero":  np.zeros((self.dim_s_extero, 1))
            }
                
        self.subs["odom"]    = rospy.Subscriber("/robot0/odom", Odometry, self.cb_odom)
        for i in range(3):
            self.sonars.append(Range())
            self.subs["sonar%d" % i]    = rospy.Subscriber("/robot0/sonar_%d" % i, Range, self.cb_range)
        self.pubs["cmd_vel"] = rospy.Publisher("/robot0/cmd_vel", Twist, queue_size = 2)

        # reset robot
        self.reset()

        self.smdict["s_proprio"] = np.zeros((self.dim_s_proprio, 1))
        self.smdict["s_extero"] = np.zeros((self.dim_s_extero, 1))

    def reset(self):
        # initialize / reset
        from stdr_msgs.srv import MoveRobot
        from geometry_msgs.msg import Pose2D
        self.srv_replace = rospy.ServiceProxy("robot0/replace", MoveRobot)
        default_pose = Pose2D()
        default_pose.x = 2.0
        default_pose.y = 2.0
        ret = self.srv_replace(default_pose)
        # print "ret", ret
        
    def step(self, x):
        if rospy.is_shutdown(): return
        self.twist.linear.x = x[0,0]  # np.random.uniform(0, 1)
        self.twist.angular.z = x[1,0] * 1.0 # np.random.uniform(-1, 1)
        self.pubs["cmd_vel"].publish(self.twist)
        # time.sleep(self.dt * 0.9)
        self.rate.sleep()

        
        # self.outdict["s_proprio"][0,0] = x[1,0]
        
        # idx = self.get_sm_index("s_extero", "vel_lin", 1)
        # self.outdict["s_extero"][idx] = x[0,0]
        
        # idx = self.get_sm_index("s_extero", "pos", 2)
        # self.outdict["s_extero"][idx] = np.array([[self.odom.pose.pose.position.x], [self.odom.pose.pose.position.y]])
        
        # idx = self.get_sm_index("s_extero", "pos", 2)
        # self.outdict["s_extero"][idx] = np.array([[self.odom.pose.pose.position.x], [self.odom.pose.pose.position.y]])

        self.outdict["s_proprio"] = self.smdict["s_proprio"]
        self.outdict["s_extero"] = self.smdict["s_extero"]
        
        return self.outdict
        
    def cb_odom(self, msg):
        # print "%s.cb_odom" % (self.__class__.__name__), type(msg)
        self.odom = msg

        self.smdict["s_proprio"][0,0] = self.odom.twist.twist.linear.x
        self.smdict["s_proprio"][1,0] = self.odom.twist.twist.angular.z
        # self.smdict["s_proprio"][1,0]  = self.odom.twist.twist.linear.x
        # self.smdict["s_extero"][self.get_sm_index("s_extero", "pos", 2)] = np.array([[self.odom.pose.pose.position.x], [self.odom.pose.pose.position.y]])
        
        # euler_angles = np.array(euler_from_quaternion([
        #     msg.pose.pose.orientation.x,
        #     msg.pose.pose.orientation.y,
        #     msg.pose.pose.orientation.z,
        #     msg.pose.pose.orientation.w
        #     ]))
        
        # self.smdict["s_extero"][self.get_sm_index("s_extero", "theta", 1)] = euler_angles[2]
        
    def cb_range(self, msg):
        # print "%s.cb_range" % (self.__class__.__name__), type(msg)
        # print "id", msg.header.frame_id
        sonar_idx = int(msg.header.frame_id.split("_")[-1])
        # print sonar_idx
        # self.get_sm_index("s_extero", "sonar", 1)
        if np.isinf(msg.range ):
            srange = 0
        else:
            srange = msg.range

        self.smdict["s_extero"][sonar_idx,0] = srange
        self.sonars[sonar_idx] = msg

# connect to lpzrobots roscontroller
class LPZBarrelSys(SMPSys):
    """LPZBarrelSys"""
    defaults = {
        'ros': True,
        'dt': 0.1,
        'dim_s_proprio': 2, # w1, w2
        'dim_s_extero': 1,
        'outdict': {},
        # 'smdict': {},
        }
    
    def __init__(self, conf = {}):
        """LPZBarrelSys.__init__

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
            self.x0 = np.zeros((self.dim_s_proprio, 1))
        self.x  = self.x0.copy()
        self.cnt = 0

        if not self.ros:
            import sys
            print "ROS not configured but this robot (%s) requires ROS, exiting" % (self.__class__.__name__)
            sys.exit(1)

        self.pubs = {
            'motors': rospy.Publisher(
                "/lpzbarrel/motors", Float64MultiArray, queue_size = 2),
            'sensors': rospy.Publisher(
                "/lpzbarrel/x", Float32MultiArray, queue_size = 2)
            }
        self.subs = {
            "/lpzbarrel/sensors": [Float64MultiArray, self.cb_sensors],
        }
        
        self.numsen_raw = 2
        self.numsen     = 2
        self.nummot     = 2
        self.sensors = Float64MultiArray()
        self.sensors.data = [0 for i in range(self.numsen_raw)]
        self.motors  = Float64MultiArray()
        self.motors.data = [0 for i in range(self.nummot)]
        self.lag = 2
            
        # timing
        self.rate = rospy.Rate(1/self.dt)
            
        # reset robot
        self.reset()

        # self.smdict["s_proprio"] = np.zeros((self.dim_s_proprio, 1))
        # self.smdict["s_extero"] = np.zeros((self.dim_s_extero, 1))

    def cb_sensors(self, msg):
        self.sensors = msg

    def prepare_inputs(self):
        inputs = np.array(self.sensors.data)
        print "%s.prepare_inputs inputs = %s" % (self.__class__.__name__, inputs)
        return inputs

    def prepare_output(self, y):
        self.motors.data = y
        # print "self.pubs", self.pubs
        self.ref.pub["_motors"].publish(self.motors)
        
    def reset(self):
        # initialize / reset
        return None
    
    def step(self, x):
        if rospy.is_shutdown(): return
        self.motors.data = x
            
        self.pubs["motors"].publish(self.motors)

        self.rate.sleep()

        # self.outdict["s_proprio"] = self.smdict["s_proprio"]
        # self.outdict["s_extero"] = self.smdict["s_extero"]
        
        return {
            's_proprio': self.prepare_inputs(),
            's_extero' : np.zeros((1,1))
            }
        
if __name__ == "__main__":
    # init node
    n = rospy.init_node("smp_sys_systems_ros")

    # get default conf
    r_conf = STDRCircularSys.defaults

    # init sys
    r = STDRCircularSys(conf = r_conf)

    print "STDR robot", r

    # run a couple of steps
    i = 0
    while not rospy.is_shutdown() and i < 100:
        # rospy.spin()
        r.step(np.random.uniform(-.3, .3, (2, 1)))
        # r.rate.sleep()
        print "step output value[%d] = %s" % (i, r.outdict)
        i += 1
