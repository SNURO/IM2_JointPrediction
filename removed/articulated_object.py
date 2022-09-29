import numpy as np
import pybullet as pb

from pytransform3d.transformations import (
    screw_axis_from_screw_parameters, transform_from_pq,
    transform_from_exponential_coordinates, pq_from_transform)


class ArticulatedObj(object):
    def __init__(self, pb, file_name, pos, orn):
        self.pb = pb #what is pb>????
        self.base_pos = pos
        self.base_orn = orn
        self.gripper2ee_pos = (-0.222, 0, 0) #change length because ur5 length is different (0.161)
        self.gripper2ee_orn = (0, 0, 0, 1)

        # Below is a manually set grasp pose (has to be changed by grasp-net output)
        self.base2gripper_pos = (-0.004, 0, 0.035)
        self.base2gripper_orn = (0, 1, 0, 1)

        #gripper T in world frame
        self.gripper_pos, self.gripper_orn = pb.multiplyTransforms(self.base_pos, self.base_orn,
                                                                   self.base2gripper_pos, self.base2gripper_orn)
        print('#################')
        print(self.gripper_pos)

        #ee T in world frame
        self.ee_pos, self.ee_orn = pb.multiplyTransforms(self.gripper_pos, self.gripper_orn,
                                                         self.gripper2ee_pos, self.gripper2ee_orn)
        print(self.ee_pos)

        self.obj_id = pb.loadURDF(file_name, pos, orn, flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                                  useFixedBase=True, globalScaling=0.1)

        # List for saving only movable joints
        self.non_fixed_joints = []

        for i in range(pb.getNumJoints(self.obj_id)):
            info = pb.getJointInfo(self.obj_id, i)
            jointID = info[0]
            # jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            # jointDamping = info[6]
            # jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            # jointMaxForce = info[10]
            # jointMaxVelocity = info[11]
            # LinkName = info[12]
            jointAxis = info[13]
            parentFramePos = info[14]
            parentFrameOrn = info[15]
            # parentIndex = info[16]

            print(jointType)

            if jointType != pb.JOINT_FIXED:
                state = pb.getLinkState(self.obj_id, jointID)  # isn't this supposed to be parent? not itself?
                linkWorldPosition = state[0]
                linkWorldOrientation = state[1]
                self.non_fixed_joints.append([jointID, jointAxis,
                                              parentFramePos, parentFrameOrn, linkWorldPosition, linkWorldOrientation,
                                              jointUpperLimit, jointLowerLimit, jointType])




        self.ee_trajectories = []
        for joint_info in self.non_fixed_joints:
            pb.setJointMotorControl2(self.obj_id, joint_info[0], pb.VELOCITY_CONTROL, targetVelocity=0, force=0.1)
            ee_trajectory = self.trajectory_creation(joint_info)
            self.ee_trajectories.append(ee_trajectory)

    def trajectory_creation(self, joint_info, visualize=True):
        jointAxis = joint_info[1]
        parentFrameOrn = joint_info[3]
        linkWorldPosition = joint_info[4]
        linkWorldOrientation = joint_info[5]
        jointUpperLimit = joint_info[6]
        jointLowerLimit = joint_info[7]
        jointType = joint_info[8]

        pq_center2joint = np.concatenate(([0, 0, 0], parentFrameOrn))  #position, quaternion
        T_center2joint = transform_from_pq(pq_center2joint)

        pq_world2center = np.concatenate(([0, 0, 0], linkWorldOrientation))
        T_world2center = transform_from_pq(pq_world2center)

        s_axis = np.matmul(np.matmul(T_world2center, T_center2joint),
                           np.concatenate((jointAxis, np.zeros(1))))[:3]  #ones? doesn't matter cause no translation Rx+p
        q = np.array(linkWorldPosition)
        """s_axis = np.matmul(T_world2center, np.concatenate((jointAxis, np.zeros(1))))[:3]"""

        print(s_axis)
        print(q)
        if jointType == self.pb.JOINT_REVOLUTE:
            h = 0.00
        elif jointType == self.pb.JOINT_PRISMATIC:
            h = float("inf")
        thetas = np.linspace(jointLowerLimit, jointUpperLimit, 5)
        A2B_list = []

        ee_transform = transform_from_pq(np.concatenate((np.array(self.ee_pos), np.array(self.ee_orn)), axis=0))
        gripper_transform = transform_from_pq(
            np.concatenate((np.array(self.gripper_pos), np.array(self.gripper_orn)), axis=0))

        for theta in thetas:
            Stheta = screw_axis_from_screw_parameters(q, s_axis, h) * theta
            A2B = transform_from_exponential_coordinates(Stheta)
            A2B_list.append(A2B)

        ee_trajectory = []

        if visualize:
            for A2B in A2B_list:
                trajectory = np.matmul(A2B, ee_transform)
                ee_trajectory.append(pq_from_transform(trajectory))
                trajectory = np.matmul(A2B, gripper_transform)
                scale = 0.02
                x_line = self.pb.addUserDebugLine(lineFromXYZ=trajectory[:3, 3],
                                                  lineToXYZ=trajectory[:3, 3] + trajectory[:3, 0] * scale,
                                                  lineColorRGB=[1, 0, 0])
                y_line = self.pb.addUserDebugLine(lineFromXYZ=trajectory[:3, 3],
                                                  lineToXYZ=trajectory[:3, 3] + trajectory[:3, 1] * scale,
                                                  lineColorRGB=[0, 1, 0])
                z_line = self.pb.addUserDebugLine(lineFromXYZ=trajectory[:3, 3],
                                                  lineToXYZ=trajectory[:3, 3] + trajectory[:3, 2] * scale,
                                                  lineColorRGB=[0, 0, 1])
        return ee_trajectory
