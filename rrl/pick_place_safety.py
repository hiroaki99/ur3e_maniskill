"""Week1 safety/contact audit for Pick-and-Place.

This module deliberately separates a raw robot-table contact candidate from a
validated unsafe-collision metric. Until a positive-control validation is done,
collision rate must remain N/A.
"""

from __future__ import annotations

import torch


class RobotTableContactAudit:
    def __init__(
        self,
        *,
        scene,
        table,
        robot_links,
        force_threshold_n: float,
        ignored_link_names=(),
        physically_validated: bool = False,
    ):
        self.scene = scene
        self.table = table
        self.robot_links = list(robot_links)
        self.force_threshold_n = float(force_threshold_n)
        self.ignored_link_names = set(ignored_link_names)
        self.physically_validated = bool(physically_validated)

    def observe(self):
        per_link = {}
        candidate_unsafe = None

        active_flags = []
        for link in self.robot_links:
            name = str(link.name)
            if name in self.ignored_link_names:
                continue

            force_vec = self.scene.get_pairwise_contact_forces(link, self.table)
            force = torch.linalg.norm(force_vec, dim=1)
            per_link[name] = force
            active_flags.append(force >= self.force_threshold_n)

        if active_flags:
            stacked = torch.stack(active_flags, dim=0)
            raw_candidate = torch.any(stacked, dim=0)
        else:
            raw_candidate = torch.zeros((1,), dtype=torch.bool)

        if self.physically_validated:
            candidate_unsafe = raw_candidate

        return {
            "raw_robot_table_contact_candidate": raw_candidate,
            "unsafe_collision": candidate_unsafe,
            "per_link_force_n": per_link,
        }
