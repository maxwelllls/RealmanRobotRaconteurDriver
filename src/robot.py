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
from general_robotics_toolbox import rpy2R, R2q


import rm_api

import os

# Simulation mode, not connecting to a real robot arm, only simulating its motion for debugging RR service
MOCK_MODE = True

# Get the current directory
curdir = os.path.dirname(os.path.abspath(__file__))

# Specify the IP address of the Realman robot
robotIp = "192.168.99.208"


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
        self.robot_info.robot_capabilities &= (
            self._robot_capabilities["jog_command"]
            & self._robot_capabilities["position_command"]
            & self._robot_capabilities["trajectory_command"]
        )
        self._trajectory_error_tol = np.deg2rad(5.0)

        # API initialization
        self._rm_api = rm_api.RealmanRobot()

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

        # Get the robot arm state
        arm_state, error = self._rm_api.get_current_arm_state(self._state_socket)
        if error:
            print("Get_Current_Arm_State error:", error)
            self.is_arm_state_thread_running = False
            return

        robot_last_recv = self._stopwatch_ellapsed_s()
        self._last_joint_state = robot_last_recv
        self._last_endpoint_state = robot_last_recv
        self._last_robot_state = robot_last_recv

        # get joint position
        joint_np = np.deg2rad(arm_state["joint"].astype(np.float64))

        # transform euler to quaternion
        pose_np = arm_state["pose"]
        R = rpy2R(pose_np[3:])
        quaternion = R2q(R)

        with self.state_lock:
            self.temp_joint_position = joint_np
            self.temp_endpoint_pose = self._node.ArrayToNamedArray(
                np.concatenate(
                    (
                        [quaternion[0], quaternion[1], quaternion[2], quaternion[3]],
                        [pose_np[0], pose_np[1], pose_np[2]],
                    )
                ),
                self._pose_dtype,
            )
            print("quaternion=", self.temp_endpoint_pose)
            self.temp_error = (
                True
                if arm_state["arm_err"] != 0 or arm_state["sys_err"] != 0
                else False
            )

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
            default_quaternion = [1, 0, 0, 0]
            default_position = [0, 0, 0]
            pose_array = np.array(
                default_quaternion + default_position, dtype=np.float64
            )
            self._endpoint_pose = self._node.ArrayToNamedArray(
                pose_array, self._pose_dtype
            )
            self._error = False
            robot_last_recv = self._stopwatch_ellapsed_s()
            self._last_joint_state = robot_last_recv
            self._last_endpoint_state = robot_last_recv
            self._last_robot_state = robot_last_recv
        else:
            # Asynchronously update the robot's state to increase the read frequency
            with self.state_lock:
                if self.temp_joint_position is not None:
                    self._joint_position = self.temp_joint_position
                    self._endpoint_pose = self.temp_endpoint_pose
                    self._error = self.temp_error
                    self.temp_joint_position = None
                    self.temp_endpoint_pose = None
                    self.temp_error = None
            threading.Thread(target=self.get_arm_state_thread).start()
        super()._run_timestep(now)

    def movej_cmd_thread(self, joint):
        try:
            self._rm_api.movej_canfd(self._cmd_socket, joint, False, 0)
        except Exception as e:
            self._error = True
            print("Error in movej_cmd_thread:", e)

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
            self._state_socket = self._rm_api.arm_socket_start(robotIp, 8080, 200)
            self._cmd_socket = self._rm_api.arm_socket_start(robotIp, 8080, 200)
            if self._state_socket is None or self._cmd_socket is None:
                print("Robot start failed")
                return
            print("Robot started")
            self._communication_failure = False

            joint_en_state, error = self._rm_api.get_joint_en_state(self._state_socket)
            if error or joint_en_state is None:
                self._communication_failure = True
                return

            # check if all joints are enabled
            self._enabled = all(joint_en_state)
            self._ready = self._enabled

        super()._start_robot()
        time.sleep(0.5)

    def close(self):
        print("Closing Robot")
        with self._lock:
            if self._running:
                if self._state_socket is not None:
                    self._rm_api.arm_socket_close(self._state_socket)
                if self._cmd_socket is not None:
                    self._rm_api.arm_socket_close(self._cmd_socket)
                self._running = False
        super().close()

    # Get the robot chain information by parsing the urdf file
    # This only needs to be run once at the beginning, and the information needs to be written to a yml file
    def print_robot_chain_from_urdf(self):
        robot_chain = rox_urdf.robot_from_xml_file(
            curdir + "/../config/rml_63.urdf",
            package=None,
            root_link="base_link",
            tip_link="link6",
        )
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

    """ The code below is untested. """

    def _send_disable(self, handler):
        if MOCK_MODE:
            self._enabled = False
            self._ready = False
            print("Robot disabled")
        else:
            for i in range(6):
                result, error = self._rm_api.set_joint_en_state(
                    self._state_socket, i, False
                )
                if error or not result:
                    print(f"Set joint {i} disable failed: {error}")
                    self._communication_failure = True
                    return
                time.sleep(0.05)
            self._enabled = False
            self._ready = False
            print("Robot disabled")

    def _send_enable(self, handler):
        if MOCK_MODE:
            self._enabled = True
            self._ready = True
            print("Robot enabled")
        else:
            for i in range(6):
                result, error = self._rm_api.set_joint_en_state(
                    self._state_socket, i, True
                )
                if error or not result:
                    print(f"Set joint {i} enable failed: {error}")
                    self._communication_failure = True
                    return
                time.sleep(0.05)
            self._enabled = True
            self._ready = True
            print("Robot enabled")

    def _send_reset_errors(self, handler):
        if not MOCK_MODE:
            result, error = self._rm_api.clear_system_err(self._state_socket)
            if error or not result:
                print(f"Clear system error failed: {error}")
                self._communication_failure = True
                return
            for i in range(6):
                result, error = self._rm_api.set_joint_clear_err(self._state_socket, i)
                if error or not result:
                    print(f"Clear joint {i} error failed: {error}")
                    self._communication_failure = True
                    return
        print("Robot reset errors")


def main():
    # Register standard Robot Raconteur service types with the node
    RRC.RegisterStdRobDefServiceTypes(RRN)

    # Read the RobotInfo yaml file to string
    robot_info_text = open(
        curdir + "/../config/realman_robot_default_config.yml", "r"
    ).read()

    # Parse the RobotInfo file
    info_loader = InfoFileLoader(RRN)
    robot_info, robot_ident_fd = info_loader.LoadInfoFileFromString(
        robot_info_text, "com.robotraconteur.robotics.robot.RobotInfo", "device"
    )

    # Create the node attributes from RobotInfo. These attributes are made available during Robot Raconteur
    # service discovery to help clients identify services
    attributes_util = AttributesUtil(RRN)
    robot_attributes = attributes_util.GetDefaultServiceAttributesFromDeviceInfo(
        robot_info.device_info
    )

    robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot")
    op_modes = robot_const["RobotOperationalMode"]

    robot_op_mode = op_modes["auto"]

    # Create the robot driver object. This object extends AbstractRobot
    robot = RealmanRobot_impl(robot_info)

    """
    Print the Robot chain information from urdf file
    This should only be run once initially, and the information should be written to the yml file manually
    Fill in the information based on other yml files before getting the correct chain information,
    otherwise initialization will fail
    """
    # robot.print_robot_chain_from_urdf()

    try:
        # Use ServerNodeSetup to initialize the server node
        with RR.ServerNodeSetup("experimental.realman_robot", 58653, argv=sys.argv):
            # Call _start_robot() to start the robot loop
            robot._start_robot()
            time.sleep(0.5)

            # Register the service and add attributes from RobotInfo file
            service_ctx = RRN.RegisterService(
                "robot", "com.robotraconteur.robotics.robot.Robot", robot
            )
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
