import socket
import json
import numpy as np


class RealmanRobot:
    def __init__(self):
        pass  # No need to initialize anything

    def arm_socket_start(self, ip, port, timeout_ms):
        try:
            # Create a socket object
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Set the timeout in seconds (convert milliseconds to seconds)
            timeout_sec = timeout_ms / 1000.0
            sock.settimeout(timeout_sec)

            # Connect to the specified IP and port
            sock.connect((ip, port))

            return sock
        except socket.error as err:
            print(f"Socket error: {err}")
            return None

    """Pass the angle to CANFD directly. If the command is correct, the robotic arm executes it immediately."""

    def movej_canfd(self, socket_fd, joint, follow, expand):
        try:
            joint = np.round(np.array(joint) * 1000).astype(int)
            cmd_str = (
                '{"command":"movej_canfd","joint":['
                + ",".join(map(str, joint))
                + '],"follow":'
                + str(follow).lower()
                + ',"expand":'
                + str(expand)
                + "}"
            )
            socket_fd.sendall(cmd_str.encode("utf-8"))
            # print(cmd_str)
        except socket.error as e:
            print(f"Socket error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def send_command(self, socket_fd, command):
        try:
            socket_fd.sendall(json.dumps(command).encode("utf-8"))
            response = socket_fd.recv(1024)
            return json.loads(response.decode("utf-8")), None
        except socket.error as e:
            return None, f"Socket error: {e}"
        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"

    def get_current_arm_state(self, socket_fd):
        command = {"command": "get_current_arm_state"}
        response, error = self.send_command(socket_fd, command)
        if error:
            return None, error

        if response.get("state") != "current_arm_state":
            return None, "Invalid response state"

        joint = np.array(response["arm_state"]["joint"]) * 0.001  # Convert to degrees
        pose = (
            np.array(response["arm_state"]["pose"]) * 0.001
        )  # Convert to meters and radians
        arm_err = response["arm_state"]["arm_err"]
        sys_err = response["arm_state"]["sys_err"]

        return {
            "joint": joint,
            "pose": pose,
            "arm_err": arm_err,
            "sys_err": sys_err,
        }, None

    def get_joint_en_state(self, socket_fd):
        command = {"command": "get_joint_en_state"}
        response, error = self.send_command(socket_fd, command)
        if error:
            return None, error
        return response.get("en_state"), None

    def set_joint_en_state(self, socket_fd, joint, state):
        command = {"command": "set_joint_en_state", "joint_en_state": [joint, state]}
        response, error = self.send_command(socket_fd, command)
        if error:
            return None, error
        return response.get("joint_en_state") == True, None

    def set_joint_clear_err(self, socket_fd, joint):
        command = {"command": "set_joint_clear_err", "joint_clear_err": joint}
        response, error = self.send_command(socket_fd, command)
        if error:
            return None, error
        return response.get("joint_clear_err") == True, None

    def clear_system_err(self, socket_fd):
        command = {"command": "clear_system_err"}
        response, error = self.send_command(socket_fd, command)
        if error:
            return None, error
        return response.get("clear_state") == True, None
