import RobotRaconteur as RR
import RobotRaconteurCompanion as RRC
RRN = RR.RobotRaconteurNode.s
import threading
import numpy as np
import time
import ctypes
import site
import sys
import drekar_launch_process
from RobotRaconteurCompanion.Util.InfoFileLoader import InfoFileLoader
from RobotRaconteurCompanion.Util.DateTimeUtil import DateTimeUtil
from RobotRaconteurCompanion.Util.SensorDataUtil import SensorDataUtil
from RobotRaconteurCompanion.Util.AttributesUtil import AttributesUtil
from contextlib import suppress
import traceback
from robotraconteur_abstract_robot import AbstractRobot
import general_robotics_toolbox as rox
import general_robotics_toolbox.urdf as rox_urdf

import os

RM_NONBLOCK = 0   # Non-blocking
RM_BLOCK = 1      # Blocking

# Simulation mode, not connecting to a real robot arm, only simulating its motion for debugging RR service
MOCK_MODE = False

# Get the current directory
curdir = os.path.dirname(os.path.abspath(__file__))

# Specify the IP address of the Realman robot
robotIp = '192.168.99.208'  

# Data structure definitions from the Realman API
class Pos(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]

class Quat(ctypes.Structure):
    _fields_ = [("w", ctypes.c_float),
                ("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float)]

class Euler(ctypes.Structure):
    _fields_ = [("rx", ctypes.c_float),
                ("ry", ctypes.c_float),
                ("rz", ctypes.c_float)]
    
class DevMsg(ctypes.Structure):
        _fields_ = [("position", Pos),
                    ("quaternion", Quat),
                    ("euler", Euler)]
            
# Robot service implementation
class RealmanRobot_impl(AbstractRobot):
    def __init__(self, robot_info):
        # Call AbstractRobot __init__
        super().__init__(robot_info, 6)
        
        # This driver does not home the robot
        self._uses_homing = False
        # Streaming position command is available
        self._has_position_command = True
        # ABB robots do not support external streaming velocity command
        self._has_velocity_command = False
        # Use a 100 Hz update loop (10 ms timestep)
        self._update_period = 0.01
        # EGM does not provide controller state or operational mode. Other drivers that use Robot Web Services (RWS)
        # do provide this information
        self._base_set_controller_state = True
        self._base_set_operational_mode = True
        # Override device capabilities from RobotInfo
        self.robot_info.robot_capabilities &= self._robot_capabilities["jog_command"] \
             & self._robot_capabilities["position_command"] & self._robot_capabilities["trajectory_command"] 
        self._trajectory_error_tol = np.deg2rad(5.)

        # API initialization
        self._rm_base = ctypes.cdll.LoadLibrary(curdir + "/libRM_Base.so.1.0.0")
        self._rm_base.RM_API_Init(632, 0)
        
        self._communication_failure = False
        
        # Time measurement variables
        self.command_count = 0
        self.prev_command_time = time.time()
        self.read_count = 0
        self.prev_read_time = time.time()

        self._enabled = True
        self._ready = True

        self.is_arm_state_thread_running = False
        self.state_lock = threading.Lock()
        self.temp_joint_position = None
        self.temp_endpoint_pose = None
        self.temp_error = None
        
        # 6 joint simulation positions, initialized to 0
        self.mock_joint_position = np.zeros((6,))
        
        self._joint_position = np.zeros((0,))
        self._endpoint_pose = np.zeros((0,), dtype=self._pose_dtype)
        
        print("RealmanRobot_impl init")
             
    def get_arm_state_thread(self):
        with self.state_lock:
            if self.is_arm_state_thread_running:
                return  # Return directly if the thread is already running
            self.is_arm_state_thread_running = True
            
        float_joint = ctypes.c_float * 6
        joint = float_joint()
        pose = DevMsg()
        arm_err = ctypes.c_uint16(1)
        sys_err = ctypes.c_uint16(1)
        
        # Get the robot arm state
        self._rm_base.Get_Current_Arm_State.argtypes = (ctypes.c_int, ctypes.c_float * 6, ctypes.POINTER(DevMsg), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16))
        self._rm_base.Get_Current_Arm_State.restype = ctypes.c_int
        ret = self._rm_base.Get_Current_Arm_State(self._state_socket, joint, pose, arm_err, sys_err)
        if ret != 0 :
            # self._communication_failure = True
            print("Get_Current_Arm_State error")
        else:
            # if (self._communication_failure == True):
            #     print("Get_Current_Arm_State ok")
            # self._communication_failure = False
            robot_last_recv = self._stopwatch_ellapsed_s()
            self._last_joint_state = robot_last_recv
            self._last_endpoint_state = robot_last_recv
            self._last_robot_state = robot_last_recv
            # Convert ctypes array to numpy array
            joint_np = np.ctypeslib.as_array(joint)

            with self.state_lock:
                # Only update to temporary variables
                self.temp_joint_position = np.deg2rad(joint_np.astype(np.float64))
                self.temp_endpoint_pose = self._node.ArrayToNamedArray(
                    np.concatenate(([pose.quaternion.w, pose.quaternion.x, pose.quaternion.y, pose.quaternion.z],
                                    [pose.position.x, pose.position.y, pose.position.z])), self._pose_dtype)
                self.temp_error = True if arm_err.value != 0 or sys_err.value != 0 else False
                
                # Print the read state frequency every second
                self.read_count += 1
                current_time = time.time()
                if current_time - self.prev_read_time >= 1: 
                    print(f"read_state_fps= {self.read_count} ")
                    self.prev_read_time = current_time
                    self.read_count = 0  

                self.is_arm_state_thread_running = False

    def _run_timestep(self, now):

        if MOCK_MODE:
            # Simulate the robot's state
            self._joint_position = self.mock_joint_position
            self._endpoint_pose = self._node.ArrayToNamedArray(np.concatenate(([0, 0, 0, 1], [0, 0, 0])) , self._pose_dtype)
            self._error = False
        else:
            # Asynchronously update the robot's state to increase the read frequency
            with self.state_lock :
                if (self.temp_joint_position is not None):
                    self._joint_position = self.temp_joint_position
                    self._endpoint_pose = self.temp_endpoint_pose
                    self._error = self.temp_error
                    self.temp_joint_position = None
                    self.temp_endpoint_pose = None
                    self.temp_error = None
            threading.Thread(target=self.get_arm_state_thread).start()
                
        super()._run_timestep(now)
       
                
    def movej_cmd_thread(self, joint):
        # ret = self._rm_base.Movej_Cmd(self._state_socket, joint, 30, 0,RM_NONBLOCK)
        ret = self._rm_base.Movej_CANFD(self._cmd_socket, joint, False, 0)
        if ret != 0:
            self._error = True

    def _send_robot_command(self, now, joint_pos_cmd, joint_vel_cmd):
        # Save position command and send it as part of _run_timestep()
        # Print the command frequency every second
        self.command_count += 1
        current_time = time.time()
        if current_time - self.prev_command_time >= 1:
            print(f"cmd_fps= {self.command_count} ")
            self.prev_command_time = current_time
            self.command_count = 0 

        # Update the robot's position command
        if joint_pos_cmd is not None:
            self._position_command = joint_pos_cmd
            
            if MOCK_MODE:
                self.mock_joint_position = joint_pos_cmd
                return

            joint = (ctypes.c_float * 6)()
            for i in range(6):
                joint[i] = np.rad2deg(joint_pos_cmd[i])
            # Send the command in a separate thread
            thread = threading.Thread(target=self.movej_cmd_thread, args=(joint,))
            thread.start()
        else:
            self._position_command = None

    def _start_robot(self):
        if MOCK_MODE:
            self._communication_failure = False
            self._enabled = True
            self._ready = True
        else:
            # Start the robot connection
            byteIP = bytes(robotIp, "gbk")
            self._state_socket = self._rm_base.Arm_Socket_Start(byteIP, 8080, 200)
            self._cmd_socket = self._rm_base.Arm_Socket_Start(byteIP, 8080, 200)
            if(self._state_socket < 0):
                print("Robot start failed")
                return
            print("Robot started")
            self._communication_failure = False
            float_joint = ctypes.c_byte * 6
            joint = float_joint()
            ret = self._rm_base.Get_Joint_EN_State(self._state_socket, joint)
            self._enabled = True
            self._ready = True
            if(ret != 0):
                self._communication_failure = True
                return
            # Check if all joints are enabled
            for i in range(6):
                if(joint[i] == 0):
                    self._enabled = False
                    self._ready = False
                    break
        super()._start_robot()
        time.sleep(0.5)

    def _send_disable(self, handler):
        if MOCK_MODE:
            self._enabled = False
            self._ready = False
            print("Robot disabled")
        # Disable all joints
        i = 0
        for i in range(6):
            ret = self._rm_base.Set_Joint_EN_State(self._state_socket, i, False, RM_BLOCK)
            if(ret == 1):
                print("Set joint " + str(i) + " disable failed")
                break
            elif (ret != 0):
                self._communication_failure = True
                print("Set joint " + str(i) + " disable failed, ret = " + str(ret))
                break
            time.sleep(0.05)
        if(i == 6):
            self._enabled = False
            self._ready = False
            print("Robot disabled")
        else:
            print("Robot disable failed")

    def _send_enable(self, handler):
        if MOCK_MODE:
            self._enabled = True
            self._ready = True
            print("Robot enabled")
        # Enable all joints
        i = 0
        for i in range(6):
            ret = self._rm_base.Set_Joint_EN_State(self._state_socket,i, True , RM_BLOCK)
            if(ret == 1):
                print("Set joint " + str(i) + " enable failed")
                break
            elif (ret != 0):
                self._communication_failure = True
                print("Set joint " + str(i) + " enable failed, ret = " + str(ret))
                break
            time.sleep(0.05)
        if(i == 6):
            self._enabled = True
            self._ready = True
            print("Robot enabled")
        else:
            self._enabled = False
            self._ready = False
            print("Robot enable failed")
            
    def _send_reset_errors(self, handler):
        ret = self._rm_base.Clear_System_Err(self._state_socket, RM_BLOCK)
        if(ret  == 1):
            print("Clear system error failed")
        elif (ret != 0):
            self._communication_failure = True
            print("Clear system error failed, ret = " + str(ret))
        i = 0
        for i in range(6):
            ret = self._rm_base.Set_Joint_Err_Clear(self._state_socket, i, RM_BLOCK)
            if(ret == 1):
                print("Clear joint " + str(i) + " error failed")
                break
            elif (ret != 0):
                self._communication_failure = True
                print("Clear joint " + str(i) + " error failed, ret = " + str(ret))
                break
            time.sleep(0.05)
        if(i == 6):
            print("Robot reset errors")
        else:
            print("Robot reset errors failed")

    def close(self):
        print("Robot closed")
        with self._lock:
            if self._running:
                if (self._state_socket >= 0 ):
                    # Close the Realman robot connection
                    self._rm_base.Arm_Socket_Close(self._state_socket)
                if (self._cmd_socket >= 0 ):
                    self._rm_base.Arm_Socket_Close(self._cmd_socket)
                self._running = False
                super().close()

    # Get the robot chain information by parsing the urdf file
    # This only needs to be run once at the beginning, and the information needs to be written to a yml file
    def print_robot_chain_from_urdf(self):
        robot_chain = rox_urdf.robot_from_xml_file(curdir + "/../config/rml_63.urdf", package=None, root_link="base_link", tip_link="link6")
        print("H:")
        for i in range(len(robot_chain.H[0])):
            print("- x: ", robot_chain.H[0][i])
            print("  y: ", robot_chain.H[1][i])
            print("  z: ", robot_chain.H[2][i])
        print("P:")
        for i in range(len(robot_chain.P[0])):
            print("- x: ", robot_chain.P[0][i])
            print("  y: ", robot_chain.P[1][i])
            print("  z: ", robot_chain.P[2][i])

def main():
    # Register standard Robot Raconteur service types with the node
    RRC.RegisterStdRobDefServiceTypes(RRN)

    # Read the RobotInfo yaml file to string 
    robot_info_text = open(curdir + "/../config/realman_robot_default_config.yml", "r").read()

    # Parse the RobotInfo file
    info_loader = InfoFileLoader(RRN)
    robot_info, robot_ident_fd = info_loader.LoadInfoFileFromString(robot_info_text, "com.robotraconteur.robotics.robot.RobotInfo", "device")
       
    # Create the node attributes from RobotInfo. These attributes are made available during Robot Raconteur 
    # service discovery to help clients identify services
    attributes_util = AttributesUtil(RRN)
    robot_attributes = attributes_util.GetDefaultServiceAttributesFromDeviceInfo(robot_info.device_info)
    
    robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot")
    op_modes = robot_const["RobotOperationalMode"]

    robot_op_mode = op_modes["auto"]

    # Create the robot driver object. This object extends AbstractRobot
    robot = RealmanRobot_impl(robot_info)
    
  
    # Print the Robot chain information from urdf file    
    # This should only be run once initially, and the information should be written to the yml file manually
    # Fill in the information based on other yml files before getting the correct chain information, 
    # otherwise initialization will fail   
    # robot.print_robot_chain_from_urdf()
            
    try:
        # Use ServerNodeSetup to initialize the server node
        with RR.ServerNodeSetup("experimental.realman_robot", 58653, argv=sys.argv):
            # Call _start_robot() to start the robot loop
            robot._start_robot()
            time.sleep(0.5)

             # Register the service and add attributes from RobotInfo file
            service_ctx = RRN.RegisterService("robot","com.robotraconteur.robotics.robot.Robot",robot)
            service_ctx.SetServiceAttributes(robot_attributes)

            drekar_launch_process.wait_exit()
            robot._close()
    except:
        # Close robot if there is an error
        with suppress(Exception):
            robot._close()
        raise
    
if __name__ == "__main__":
    main()
