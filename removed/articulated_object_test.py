#!/usr/bin/env python3
"""Example script for motion planning with a robot arm in pybullet."""

import pybullet as pb
import logging
import time

from articulated_object import ArticulatedObj


def main():
    # Parameters for running this example
    delay: float = 0.01
    log_level: str = 'WARN'

    # Configure logging
    logging.root.setLevel(log_level)
    logging.basicConfig()

    # Creating a simulation client
    physicsClient = pb.connect(pb.GUI)

    base_pos = (0.5, 0, 0.05)  # x y z
    base_orn = (0, 0, 0, 1)  # quaternion
    artObj = ArticulatedObj(pb, "./148/mobility.urdf", base_pos, base_orn)  # revolute joint
    #artObj = ArticulatedObj(pb, "./27619/mobility.urdf", base_pos, base_orn)  # prismatic joint
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
    while True:
        pb.stepSimulation()
        time.sleep(delay)
    pb.disconnect()


if __name__ == '__main__':
    main()
