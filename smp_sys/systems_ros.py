"""smp_sys

ROS based systems, simulators and real robots

These are different from simple systems such as a single python
function in that they require interaction with the outside world.

The basic pattern is that all system state variables are class members
and the message callbacks write to these variables asynchronously,
whenever they are called. The synchronous loop then just uses the
current variable value.

The step function
 1. takes the motor input from the outside
 2. sets the actual motor values applying scaling etc as required
 3. publishes the motor values via ROS
 4. sleeps for one rate cycle
 5. fetches the sensor feedback as the current value of corresponding
    state variables and returns them

 Simulators: Simple Two-Dimensional Robot Simulator (STDR), lpzrobots, MORSE, (Webots, Gazebo, ...)
Real robots: Sphero, HUCar, Turtlebot, Quadrotor, Puppy, Nao
"""

import time
import numpy as np
from collections import OrderedDict

# ROS imports
try:
    import rospy
    from tf.transformations import euler_from_quaternion, quaternion_from_euler
except Exception, e:
    print "Import rospy failed with %s" % (e, )

from std_msgs.msg      import Float64MultiArray, Float32MultiArray
from std_msgs.msg      import Float32, ColorRGBA

from nav_msgs.msg      import Odometry
from geometry_msgs.msg import Twist, Quaternion #, Point, Pose, TwistWithCovariance, Vector3
from sensor_msgs.msg   import Range
from sensor_msgs.msg   import Imu

# smpsys base class
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
        'dt': 0.01,
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
            'sensors': rospy.Subscriber("/lpzbarrel/sensors", Float64MultiArray, self.cb_sensors),
        }
        
        self.numsen_raw = 2
        self.numsen     = 2
        self.nummot     = 2
        self.sensors = Float64MultiArray()
        self.sensors.data = [0.0 for i in range(self.numsen_raw)]
        self.motors  = Float64MultiArray()
        self.motors.data = [0.0 for i in range(self.nummot)]
        self.lag = 2
            
        # timing
        self.rate = rospy.Rate(1/self.dt)
            
        # reset robot
        self.reset()

        # self.smdict["s_proprio"] = np.zeros((self.dim_s_proprio, 1))
        # self.smdict["s_extero"] = np.zeros((self.dim_s_extero, 1))

    def cb_sensors(self, msg):
        # print "%s.cb_sensors msg = %s" % (self.__class__.__name__, msg)
        self.sensors = msg

    def prepare_inputs(self):
        sdata = self.sensors.data
        print "%s.prepare_inputs sdata = %s" % (self.__class__.__name__, type(sdata))
        inputs = np.array([sdata])
        print "%s.prepare_inputs inputs = %s" % (self.__class__.__name__, inputs.shape)
        return inputs.T

    def prepare_output(self, y):
        self.motors.data = y
        # print "self.pubs", self.pubs
        self.ref.pub["_motors"].publish(self.motors)
        
    def reset(self):
        # initialize / reset
        return None
    
    def step(self, x):
        if rospy.is_shutdown(): return
        print "%s.step x = %s %s, motor data = %s" % (self.__class__.__name__, x.dtype, x.flatten().tolist(), type(self.motors.data))

        self.motors.data = x.flatten().tolist()
            
        self.pubs["motors"].publish(self.motors)

        self.rate.sleep()

        # self.outdict["s_proprio"] = self.smdict["s_proprio"]
        # self.outdict["s_extero"] = self.smdict["s_extero"]
        
        return {
            's_proprio': self.prepare_inputs(),
            's_extero' : np.zeros((1,1))
            }

# class robotSphero(robot): from smp_sphero/hk2.py
class SpheroSys(SMPSys):
    """SpheroSys

    Sphero ROS based system

    FIXME: control modes
    """
    defaults = {
        'ros': True,
        'dt': 0.05,
        'dim_s_proprio': 2, # linear, angular
        'dim_s_extero': 1,
        'outdict': {},
        'control': 'twist',
    }

    # def __init__(self, ref):
    #     robot.__init__(self, ref)
    def __init__(self, conf = {}):
        """SpheroSys.__init__

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

        # self.pubs = {
        #     'motors': rospy.Publisher(
        #         "/lpzbarrel/motors", Float64MultiArray, queue_size = 2),
        #     'sensors': rospy.Publisher(
        #         "/lpzbarrel/x", Float32MultiArray, queue_size = 2)
        #     }
        # self.subs = {
        #     'sensors': rospy.Subscriber("/lpzbarrel/sensors", Float64MultiArray, self.cb_sensors),
        # }
        
        self.pubs = {
            '_cmd_vel':     rospy.Publisher("/cmd_vel",     Twist, queue_size = 2),
            '_cmd_vel_raw': rospy.Publisher("/cmd_vel_raw", Twist, queue_size = 2),
            '_set_color':   rospy.Publisher("/set_color",   ColorRGBA, queue_size = 2),
            '_lpzros_x':    rospy.Publisher("/lpzros/x",    Float32MultiArray, queue_size = 2)
            }
        # self.cb = {
        #     "/imu": get_cb_dict(self.cb_imu),
        #     "/odom": get_cb_dict(self.cb_odom),
        #     }
        
        self.subs = {
            'imu':  rospy.Subscriber("/imu", Imu, self.cb_imu),
            'odom': rospy.Subscriber("/odom", Odometry, self.cb_odom),
            }

        # timing
        self.rate = rospy.Rate(1/self.dt)

        self.cb_imu_cnt  = 0
        self.cb_odom_cnt = 0
        
        # custom
        self.numsen_raw = 10 # 8 # 5 # 2
        self.numsen     = 10 # 8 # 5 # 2
        
        self.imu  = Imu()
        self.odom = Odometry()
        # sphero color
        self.color = ColorRGBA()
        self.motors = Twist()
        
        self.msg_inputs     = Float32MultiArray()
        self.msg_motors     = Float64MultiArray()
        self.msg_sensor_exp = Float64MultiArray()

        self.imu_lin_acc_gain = 0 # 1e-1
        self.imu_gyrosco_gain = 1e-1
        self.imu_orienta_gain = 0 # 1e-1
        self.linear_gain      = 0 # 1.0 # 1e-1
        self.pos_gain         = 0 # 1e-2
        self.output_gain = 255 # 120 # 120
        
        # sphero lag is 4 timesteps
        self.lag = 1 # 2
        
    def cb_imu(self, msg):
        """SpheroSys.cb_imu

        ROS IMU sensor callback: use odometry and incoming imu data to trigger
        sensorimotor loop execution
        """
        # FIXME: do the averaging here
        self.imu = msg
        imu_vec_acc = np.array((self.imu.linear_acceleration.x, self.imu.linear_acceleration.y, self.imu.linear_acceleration.z))
        imu_vec_gyr = np.array((self.imu.angular_velocity.x, self.imu.angular_velocity.y, self.imu.angular_velocity.z))
        (r, p, y) = tf.transformations.euler_from_quaternion([self.imu.orientation.x, self.imu.orientation.y, self.imu.orientation.z, self.imu.orientation.w])
        imu_vec_ori = np.array((r, p, y))
        imu_vec_ = np.hstack((imu_vec_acc, imu_vec_gyr, imu_vec_ori)).reshape(self.imu_vec.shape)
        self.imu_vec = self.imu_vec * self.imu_smooth + (1 - self.imu_smooth) * imu_vec_
        # print "self.imu_vec", self.imu_vec

        self.cb_imu_cnt += 1
        
    def cb_odom(self, msg):
        """ROS odometry callback, copy incoming data into local memory"""
        # print type(msg)
        self.odom = msg        
        self.cb_odom_cnt += 1

    def prepare_inputs_all(self):
        inputs = (self.odom.twist.twist.linear.x * self.linear_gain, self.odom.twist.twist.linear.y * self.linear_gain,
                         self.imu_vec[0] * self.imu_lin_acc_gain,
                         self.imu_vec[1] * self.imu_lin_acc_gain,
                         self.imu_vec[2] * self.imu_lin_acc_gain,
                         self.imu_vec[3] * self.imu_gyrosco_gain,
                         self.imu_vec[4] * self.imu_gyrosco_gain,
                         self.imu_vec[5] * self.imu_gyrosco_gain,
                         self.odom.pose.pose.position.x * self.pos_gain,
                         self.odom.pose.pose.position.y * self.pos_gain,
                         )
        print "%s.prepare_inputs_all inputs = %s" % (self.__class__.__name__, inputs)
        return np.array([inputs])

    def prepare_inputs(self):
        inputs = (self.odom.twist.twist.linear.x * self.linear_gain, self.odom.twist.twist.linear.y * self.linear_gain)
        print "%s.prepare_inputs inputs = %s" % (self.__class__.__name__, inputs)
        return np.array([inputs])

    def prepare_output(self, y):
        # self.motors.linear.x = y[0,0] * self.output_gain
        # self.motors.linear.y = y[1,0] * self.output_gain
        # self.pub["_cmd_vel"].publish(self.motors)
        self.motors.linear.x  = y[1,0] * self.output_gain * 1.414 # ?
        self.motors.angular.z = y[0,0] * 1 # self.output_gain

    def step(self, x):
        print "x", x
        # x_ = self.prepare_inputs()

        self.prepare_output(x)
        
        self.pubs["_cmd_vel_raw"].publish(self.motors)
        print "%s.prepare_output y = %s , motors = %s" % (y, self.motors)

        self.rate.sleep()
        
        return {
            's_proprio': self.prepare_inputs(),
            's_extero' : np.zeros((1,1))
            }
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type = str, default = "stdr", help = "Which system to test [stdr] from stdr, lpzbarrel, ...")
    args = parser.parse_args()
    
    # init node
    n = rospy.init_node("smp_sys_systems_ros")

    if args.system == "stdr":
        syscls = STDRCircularSys
    elif args.system == "lpzbarrel":
        syscls = LPZBarrelSys
    elif args.system == "sphero":
        syscls = SpheroSys
        
    # get default conf
    r_conf = syscls.defaults

    # init sys
    r = syscls(conf = r_conf)

    print "%s robot = %s" % (args.system, r)

    # run a couple of steps
    i = 0
    while not rospy.is_shutdown() and i < 100:
        # rospy.spin()
        x = r.step(np.random.uniform(-.3, .3, (r.dim_s_proprio, 1)))
        # r.rate.sleep()
        print "step[%d] output = %s" % (i, x)
        i += 1
