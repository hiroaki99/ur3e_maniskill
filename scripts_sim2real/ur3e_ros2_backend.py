#!/usr/bin/env python3

import json
import socket
import numpy as np
import time

from scripts_sim2real.robot_backend import (
    RobotBackend,
    TCPPose,
)


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def wrap_to_pi(angle):
    return float(
        np.arctan2(
            np.sin(angle),
            np.cos(angle),
        )
    )


class UR3eROS2Backend(RobotBackend):

    def __init__(
        self,
        host="127.0.0.1",
        port=50051,
        timeout=2.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout

    def _request_payload(self, request):

        with socket.create_connection(
            (self.host, self.port),
            timeout=self.timeout,
        ) as sock:

            sock.sendall(
                (
                    json.dumps(request)
                    + "\n"
                ).encode("utf-8")
            )

            file_obj = sock.makefile(
                "r",
                encoding="utf-8",
            )

            line = file_obj.readline()

        if not line:
            raise RuntimeError(
                "ROS2 bridge returned an empty response."
            )

        response = json.loads(line)

        if not response.get("ok", False):
            raise RuntimeError(
                response.get(
                    "error",
                    "Unknown bridge error",
                )
            )

        return response["result"]


    def _request(self, command):

        return self._request_payload(
            {
                "command": command,
            }
        )


    def ping(self):
        return self._request("ping")

    def wait_until_ready(
        self,
        timeout_sec=10.0,
        poll_interval_sec=0.1,
    ):
        """
        Bridgeが /joint_states と TCP pose の両方を
        受信するまで待つ。

        Returns:
            True

        Raises:
            RuntimeError:
                timeoutまでにROS状態が揃わなかった場合。
        """

        deadline = (
            time.monotonic()
            + timeout_sec
        )

        last_joint_error = None
        last_tcp_error = None

        while time.monotonic() < deadline:

            joint_ready = False
            tcp_ready = False

            try:
                self._request(
                    "get_joint_positions"
                )
                joint_ready = True

            except RuntimeError as exc:
                last_joint_error = str(exc)

            try:
                self._request(
                    "get_tcp_pose"
                )
                tcp_ready = True

            except RuntimeError as exc:
                last_tcp_error = str(exc)

            if joint_ready and tcp_ready:
                return True

            time.sleep(
                poll_interval_sec
            )

        raise RuntimeError(
            "ROS2 bridge did not become ready "
            f"within {timeout_sec:.1f} s. "
            f"joint_state_error={last_joint_error!r}, "
            f"tcp_pose_error={last_tcp_error!r}"
        )

    def get_joint_positions(self):
        raw = self._request(
            "get_joint_positions"
        )

        result = {}

        for name in JOINT_NAMES:
            if name not in raw:
                raise RuntimeError(
                    f"Joint {name} was not returned "
                    "by ROS2 bridge."
                )

            # ManiSkillとの比較で扱いやすいよう
            # [-pi, pi] に正規化
            result[name] = wrap_to_pi(
                raw[name]
            )

        return result

    def get_tcp_pose(self):
        data = self._request(
            "get_tcp_pose"
        )

        return TCPPose(
            frame_id=data["frame_id"],
            position=[
                float(data["position"]["x"]),
                float(data["position"]["y"]),
                float(data["position"]["z"]),
            ],
            orientation_xyzw=[
                float(
                    data[
                        "orientation_xyzw"
                    ]["x"]
                ),
                float(
                    data[
                        "orientation_xyzw"
                    ]["y"]
                ),
                float(
                    data[
                        "orientation_xyzw"
                    ]["z"]
                ),
                float(
                    data[
                        "orientation_xyzw"
                    ]["w"]
                ),
            ],
        )


    def preview_joint_positions(
        self,
        joint_positions,
    ):
        request = {
            "command":
                "preview_joint_positions",

            "joint_positions": {
                name: float(value)
                for name, value
                in joint_positions.items()
            },
        }

        return self._request_payload(
            request
        )


    def command_joint_positions(
        self,
        joint_positions,
        duration_sec=2.0,
    ):
        request = {
            "command": "command_joint_positions",
            "joint_positions": {
                name: float(value)
                for name, value
                in joint_positions.items()
            },
            "duration_sec": float(
                duration_sec
            ),
        }

        return self._request_payload(
            request
        )

    def close(self):
        # 現在はリクエストごとにsocketを閉じるので
        # 特別な終了処理は不要
        pass