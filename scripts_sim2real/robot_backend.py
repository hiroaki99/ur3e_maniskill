#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class TCPPose:
    frame_id: str
    position: Sequence[float]
    orientation_xyzw: Sequence[float]


class RobotBackend(ABC):

    @abstractmethod
    def get_joint_positions(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_tcp_pose(self) -> TCPPose:
        raise NotImplementedError

    @abstractmethod
    def command_joint_positions(
        self,
        joint_positions: Dict[str, float],
        duration_sec: float = 2.0,
    ):
        """
        指定関節角へ移動する。

        joint_positions:
            joint name -> target angle [rad]

        duration_sec:
            目標到達時間 [s]
        """
        raise NotImplementedError

    def close(self):
        pass