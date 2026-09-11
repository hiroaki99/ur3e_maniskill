#!/usr/bin/env python3

import argparse
import json
import socketserver
import threading

import math
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


ACTION_NAME = (
    "/scaled_joint_trajectory_controller/"
    "follow_joint_trajectory"
)

GOAL_POSITION_TOLERANCE_RAD = 0.001

GOAL_TIME_TOLERANCE_SEC = 0.5


# 初期検証用の安全制限。
# 1回のcommandで各関節最大 ±0.05 rad（約2.9°）まで。
MAX_JOINT_DELTA_RAD_BY_JOINT = {
    "shoulder_pan_joint": math.radians(21.0),
    "shoulder_lift_joint": 0.05,
    "elbow_joint": 0.05,
    "wrist_1_joint": 0.05,
    "wrist_2_joint": 0.05,
    "wrist_3_joint": 0.05,
}

class RobotStateStore:
    """
    ROS callbacks と TCP server 間で共有する最新状態。
    """

    def __init__(self):
        self.lock = threading.Lock()

        self.joint_positions = None
        self.tcp_pose = None

    def update_joint_state(self, msg):
        joint_positions = {
            name: float(position)
            for name, position in zip(
                msg.name,
                msg.position,
            )
        }

        with self.lock:
            self.joint_positions = joint_positions

    def update_tcp_pose(self, msg):
        pose = msg.pose

        tcp_pose = {
            "frame_id": msg.header.frame_id,
            "position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            "orientation_xyzw": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w),
            },
        }

        with self.lock:
            self.tcp_pose = tcp_pose

    def get_joint_positions(self):
        with self.lock:
            if self.joint_positions is None:
                return None

            return dict(self.joint_positions)

    def get_tcp_pose(self):
        with self.lock:
            if self.tcp_pose is None:
                return None

            return dict(self.tcp_pose)

    def is_ready(self):
        with self.lock:
            return (
                self.joint_positions is not None
                and self.tcp_pose is not None
            )

def nearest_equivalent_angle(
    target,
    reference,
):
    """
    target + 2*pi*k の中から、
    現在値 reference に最も近い角度を返す。

    例:
      target    = 0.0078
      reference = 18.8573

    -> 約18.8573を返す。
    """

    two_pi = 2.0 * math.pi

    k = round(
        (reference - target)
        / two_pi
    )

    return (
        target
        + k * two_pi
    )

class UR3eStateNode(Node):

    def __init__(self, store):
        super().__init__("ur3e_sim2real_bridge")

        self.store = store

        self.joint_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            qos_profile_sensor_data,
        )

        self.tcp_subscription = self.create_subscription(
            PoseStamped,
            "/tcp_pose_broadcaster/pose",
            self.tcp_pose_callback,
            qos_profile_sensor_data,
        )

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            ACTION_NAME,
        )

        self.get_logger().info(
            "UR3e ROS2 bridge subscriptions initialized."
        )

        self.get_logger().info(
            f"Trajectory action: {ACTION_NAME}"
        )

    def joint_state_callback(self, msg):
        self.store.update_joint_state(msg)

    def tcp_pose_callback(self, msg):
        self.store.update_tcp_pose(msg)

    def resolve_joint_targets(
        self,
        target_positions,
    ):
        """
        canonical joint target を、
        現在の実機raw joint angleに最も近い
        2*pi等価角へ変換する。

        この関数自体はロボットを動かさない。
        """

        current_raw = (
            self.store.get_joint_positions()
        )

        if current_raw is None:
            raise RuntimeError(
                "No joint state available."
            )

        resolved_targets = {}
        delta_rad = {}

        for name in JOINT_NAMES:

            if name not in target_positions:
                raise RuntimeError(
                    f"Missing target joint: {name}"
                )

            if name not in current_raw:
                raise RuntimeError(
                    f"Missing current joint: {name}"
                )

            target = float(
                target_positions[name]
            )

            current = float(
                current_raw[name]
            )

            target_near = (
                nearest_equivalent_angle(
                    target,
                    current,
                )
            )

            delta = (
                target_near
                - current
            )

            limit = MAX_JOINT_DELTA_RAD_BY_JOINT[name]

            if abs(delta)> limit:
                raise RuntimeError(
                    f"Safety limit exceeded for "
                    f"{name}: "
                    f"current={current:.6f}, "
                    f"target={target_near:.6f}, "
                    f"delta={delta:.6f} rad, "
                    f"limit="
                    f"{limit:.6f} rad"
                )

            resolved_targets[name] = (
                target_near
            )

            delta_rad[name] = delta

        return {
            "current_raw": {
                name: float(
                    current_raw[name]
                )
                for name in JOINT_NAMES
            },
            "resolved_targets": {
                name: float(
                    resolved_targets[name]
                )
                for name in JOINT_NAMES
            },
            "delta_rad": {
                name: float(
                    delta_rad[name]
                )
                for name in JOINT_NAMES
            },
        }

    def command_joint_positions(
        self,
        target_positions,
        duration_sec,
        timeout_sec=10.0,
    ):
        resolution = (
            self.resolve_joint_targets(
                target_positions
            )
        )

        resolved_targets = (
            resolution[
                "resolved_targets"
            ]
        )


        goal = FollowJointTrajectory.Goal()

        goal.trajectory.joint_names = (
            JOINT_NAMES.copy()
        )


        point = JointTrajectoryPoint()

        point.positions = [
            resolved_targets[name]
            for name in JOINT_NAMES
        ]


        sec = int(duration_sec)

        nanosec = int(
            (duration_sec - sec)
            * 1_000_000_000
        )

        point.time_from_start = Duration(
            sec=sec,
            nanosec=nanosec,
        )

        goal.trajectory.points = [
            point
        ]


        goal.goal_tolerance = []

        for name in JOINT_NAMES:
            tolerance = JointTolerance()

            tolerance.name = name

            tolerance.position = (
                GOAL_POSITION_TOLERANCE_RAD
            )

            # velocity / acceleration は
            # Controllerの既定値を使用
            tolerance.velocity = 0.0
            tolerance.acceleration = 0.0

            goal.goal_tolerance.append(
                tolerance
            )


        goal.goal_time_tolerance = Duration(
            sec=GOAL_TIME_TOLERANCE_SEC,
            nanosec=0,
        )


        # --------------------------------------------
        # Goal送信
        # --------------------------------------------

        goal_event = threading.Event()

        goal_holder = {}

        send_future = (
            self.trajectory_client.send_goal_async(
                goal
            )
        )


        

        def on_goal_response(future):
            try:
                goal_holder["goal_handle"] = (
                    future.result()
                )
            except Exception as exc:
                goal_holder["exception"] = exc
            finally:
                goal_event.set()

        send_future.add_done_callback(
            on_goal_response
        )

        if not goal_event.wait(timeout_sec):
            raise RuntimeError(
                "Timed out waiting for "
                "trajectory goal response."
            )

        if "exception" in goal_holder:
            raise goal_holder["exception"]

        goal_handle = (
            goal_holder["goal_handle"]
        )

        if not goal_handle.accepted:
            raise RuntimeError(
                "Trajectory goal was rejected."
            )


        # --------------------------------------------
        # Result待機
        # --------------------------------------------

        result_event = threading.Event()

        result_holder = {}

        result_future = (
            goal_handle.get_result_async()
        )

        def on_result(future):
            try:
                result_holder["result"] = (
                    future.result()
                )
            except Exception as exc:
                result_holder["exception"] = exc
            finally:
                result_event.set()

        result_future.add_done_callback(
            on_result
        )

        if not result_event.wait(timeout_sec):
            raise RuntimeError(
                "Timed out waiting for "
                "trajectory result."
            )

        if "exception" in result_holder:
            raise result_holder["exception"]


        wrapped_result = (
            result_holder["result"]
        )

        result = wrapped_result.result


        success = (
            result.error_code == 0
        )

        return {
            "success": success,
            "error_code": int(
                result.error_code
            ),
            "error_string": str(
                result.error_string
            ),
            "resolved_targets": {
                name: float(
                    resolved_targets[name]
                )
                for name in JOINT_NAMES
            },
        }


class BridgeRequestHandler(socketserver.StreamRequestHandler):

    def handle(self):
        line = self.rfile.readline()

        if not line:
            return

        try:
            request = json.loads(
                line.decode("utf-8")
            )

            command = request.get("command")

            if command == "ping":
                response = {
                    "ok": True,
                    "result": "pong",
                }

            elif command == "get_joint_positions":
                data = self.server.state_store.get_joint_positions()

                if data is None:
                    response = {
                        "ok": False,
                        "error": (
                            "No /joint_states message "
                            "has been received yet."
                        ),
                    }
                else:
                    response = {
                        "ok": True,
                        "result": data,
                    }

            elif command == "get_tcp_pose":
                data = self.server.state_store.get_tcp_pose()

                if data is None:
                    response = {
                        "ok": False,
                        "error": (
                            "No /tcp_pose_broadcaster/pose "
                            "message has been received yet."
                        ),
                    }
                else:
                    response = {
                        "ok": True,
                        "result": data,
                    }

            elif command == "preview_joint_positions":

                joint_positions = request.get(
                    "joint_positions"
                )

                if not isinstance(
                    joint_positions,
                    dict,
                ):
                    raise RuntimeError(
                        "joint_positions must be "
                        "a dict."
                    )

                result = (
                    self.server.ros_node
                    .resolve_joint_targets(
                        joint_positions
                    )
                )

                response = {
                    "ok": True,
                    "result": result,
                }

            elif command == "command_joint_positions":

                joint_positions = request.get(
                    "joint_positions"
                )

                duration_sec = float(
                    request.get(
                        "duration_sec",
                        2.0,
                    )
                )

                if not isinstance(
                    joint_positions,
                    dict,
                ):
                    raise RuntimeError(
                        "joint_positions must be a dict."
                    )

                result = (
                    self.server.ros_node.command_joint_positions(
                        joint_positions,
                        duration_sec,
                    )
                )

                response = {
                    "ok": True,
                    "result": result,
                }

            elif command == "is_ready":

                response = {
                    "ok": True,
                    "result": (
                        self.server.state_store.is_ready()
                    ),
                }

            else:
                response = {
                    "ok": False,
                    "error": f"Unknown command: {command}",
                }

        except Exception as exc:
            response = {
                "ok": False,
                "error": repr(exc),
            }

        payload = (
            json.dumps(response)
            + "\n"
        ).encode("utf-8")

        self.wfile.write(payload)


class ThreadingBridgeServer(
    socketserver.ThreadingTCPServer
):
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        handler_class,
        state_store,
        ros_node,
    ):
        super().__init__(
            server_address,
            handler_class,
        )

        self.state_store = state_store
        self.ros_node = ros_node

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=50051,
    )

    args = parser.parse_args()

    rclpy.init()

    state_store = RobotStateStore()

    node = UR3eStateNode(
        state_store
    )

    # --------------------------------------------------------
    # ROS executor thread
    # --------------------------------------------------------

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )

    server = None

    try:
        # ----------------------------------------------------
        # まずTCP portをbindする。
        #
        # bind失敗時にはROS spin threadを起動しない。
        # ----------------------------------------------------

        server = ThreadingBridgeServer(
            (args.host, args.port),
            BridgeRequestHandler,
            state_store,
            node,
        )

        # ----------------------------------------------------
        # bind成功後にROS callback処理を開始
        # ----------------------------------------------------

        ros_thread.start()

        print(
            f"[INFO] UR3e ROS2 bridge listening on "
            f"{args.host}:{args.port}"
        )

        server.serve_forever()

    except KeyboardInterrupt:
        print(
            "\n[INFO] Shutting down bridge."
        )

    finally:
        # ----------------------------------------------------
        # TCP server cleanup
        # ----------------------------------------------------

        if server is not None:
            server.server_close()

        # ----------------------------------------------------
        # rclpy.spin() を終了させる
        # ----------------------------------------------------

        if rclpy.ok():
            rclpy.shutdown()

        # ----------------------------------------------------
        # ROS thread終了待ち
        # ----------------------------------------------------

        if ros_thread.is_alive():
            ros_thread.join(
                timeout=2.0
            )

        # ----------------------------------------------------
        # ROS node cleanup
        # ----------------------------------------------------

        node.destroy_node()


if __name__ == "__main__":
    main()