import numpy as np
import pybullet as pb
import grasp_choosing
import math
import os
import sys
import logging

# note that below pytransform3d all treats quaternions (w,x,y,z) whereas pybullet treats them (x,y,z,w)
from pytransform3d.transformations import (
    screw_axis_from_screw_parameters, transform_from_pq,
    transform_from_exponential_coordinates, pq_from_transform)

MOTION_LIB_PATH = './motion-planners'   # if there is motion-planners in different directory
sys.path.append(MOTION_LIB_PATH)
from motion_planners.rrt_connect import birrt

#pytransform, pybullet quaternion representation is different, pb: [p, x,y,z, w], pytransform: [p, w, x,y,z]
#pbpq2pq is needed to be called before visualization

def frame_convert(orn):
    x = np.array([[0], [0], [1]])  # hardcoded Rot (graspnet -> actual URDF/robot ee) 
    y = np.array([[1], [0], [0]])
    z = np.array([[0], [1], [0]])

    R_Grasp2URDF = np.concatenate((x, y, z), axis=1)

    return orn @ R_Grasp2URDF

class trajectory(object):
    def __init__(self, pb, obj_id, base_pos, base_orn, grasp_pos, grasp_orn, jointinfo_list, currentangle):
        self.pb = pb
        # input orn is xyzw (in form of pybullet)
        self.obj_id = obj_id
        self.base_pos = base_pos
        self.base_orn = base_orn
        self.gripper2ee_pos = (0, 0, -0.222) #change length because ur5 length is different (0.161)
        self.gripper2ee_orn = (0,0,0,1)
        self.currentangle = currentangle

        self.worldbase2base_pos = base_pos
        self.worldbase2base_orn = pb.getQuaternionFromEuler((math.pi/2,0,-math.pi/2 ))
        self.base2gripper_pos = grasp_pos
        self.base2gripper_orn = grasp_orn

        # gripper T in world frame
        self.gripper_pos, self.gripper_orn = pb.multiplyTransforms(self.worldbase2base_pos, self.worldbase2base_orn,
                                                                   self.base2gripper_pos, self.base2gripper_orn)
        #ee T in world frame
        self.ee_pos, self.ee_orn = pb.multiplyTransforms(self.gripper_pos, self.gripper_orn,
                                                         self.gripper2ee_pos, self.gripper2ee_orn)


        self.ee_trajectories_up = []    # all in form of vec7, wxyz of pytransform
        self.ee_trajectories_down =[]   # up until upper joint limit, down until lower joint limit
        self.gripper_trajectories_up = []
        self.gripper_trajectories_down = []

        for joint_info in jointinfo_list:
            pb.setJointMotorControl2(self.obj_id, joint_info[0], pb.VELOCITY_CONTROL, targetVelocity=0, force=0.1)
            noise=[0,0,0]   # no noise added
            #noise = np.random.normal(0, 0.3, 3)
            ee_trajectory_up, gripper_trajectory_up = self.trajectory_creation(joint_info, visualize=False, upwards=True, noise=noise)
            ee_trajectory_down, gripper_trajectory_down = self.trajectory_creation(joint_info, visualize=False, upwards=False, noise=noise)
            self.ee_trajectories_up= ee_trajectory_up
            self.gripper_trajectories_up= gripper_trajectory_up
            self.ee_trajectories_down= ee_trajectory_down
            self.gripper_trajectories_down= gripper_trajectory_down
            

    def trajectory_creation(self, joint_info, visualize=True, upwards=True, noise=[0,0,0]):
        jointAxis = joint_info[1]
        parentFrameOrn = joint_info[3]
        linkWorldPosition = joint_info[4]
        linkWorldOrientation = joint_info[5]
        jointUpperLimit = joint_info[6]
        jointLowerLimit = joint_info[7]
        currentangle = joint_info[8]
        jointType = joint_info[9]

        pq_world2child = pbpq2pq(np.concatenate(([0, 0, 0], linkWorldOrientation)))
        T_world2child = transform_from_pq(pq_world2child)

        s_axis = np.matmul(T_world2child, np.concatenate((jointAxis, np.ones(1))))[:3] + noise # added noise
        q = np.array(linkWorldPosition)

        if jointType == self.pb.JOINT_REVOLUTE:
            h = 0.00
        elif jointType == self.pb.JOINT_PRISMATIC:
            h = float("inf")

        if upwards:     # below 5 can be modified to choose how many intermidiate SE(3) poses
            thetas = np.linspace(currentangle, jointUpperLimit, 5)
        else:
            thetas = np.linspace(currentangle, jointLowerLimit, 5)

        A2B_list = []

        # change pq format to SE(3)
        ee_transform = transform_from_pq(pbpq2pq(np.concatenate((np.array(self.ee_pos), np.array(self.ee_orn)), axis=0)))
        gripper_transform = transform_from_pq(
            pbpq2pq(np.concatenate((np.array(self.gripper_pos), np.array(self.gripper_orn)), axis=0)))

        for theta in thetas:
            Stheta = screw_axis_from_screw_parameters(q, s_axis, h) * theta
            A2B = transform_from_exponential_coordinates(Stheta)
            A2B_list.append(A2B)

        ee_trajectory = []
        gripper_trajectory = []

        for A2B in A2B_list:
            trajectory = np.matmul(A2B, ee_transform)   # T of ee
            ee_trajectory.append(pq_from_transform(trajectory))         # trajectory of end effector, quaternion, pwxyz in format of pytransform
            trajectory = np.matmul(A2B, gripper_transform)
            gripper_trajectory.append(pq_from_transform(trajectory))

        return ee_trajectory, gripper_trajectory

    def visualization(self, trajectories):
        # works correct in trajectory format of vec7, [p, w, x, y, z]
        scale = 0.01
        for xyz in trajectories:
            xyz_temp = transform_from_pq(xyz)
            x_line = self.pb.addUserDebugLine(lineFromXYZ=xyz_temp[:3, 3],
                                                    lineToXYZ=xyz_temp[:3, 3] + xyz_temp[:3, 0] * scale,
                                                    lineColorRGB=[1, 0, 0])
            y_line = self.pb.addUserDebugLine(lineFromXYZ=xyz_temp[:3, 3],
                                                    lineToXYZ=xyz_temp[:3, 3] + xyz_temp[:3, 1] * scale,
                                                    lineColorRGB=[0, 1, 0])
            z_line = self.pb.addUserDebugLine(lineFromXYZ=xyz_temp[:3, 3],
                                                    lineToXYZ=xyz_temp[:3, 3] + xyz_temp[:3, 2] * scale,
                                                    lineColorRGB=[0, 0, 1])
            
    def frame_change(self, graspnet_orn):    # orientation assumed to be z toward object, x toward two fingers   
                                         # change that to orientation defined in robot URDF (T current->URDF target)
                                         # multiplied afterwards ORN saved in this class
                                         # xyzw (0.5 0.5 0.5 -0.5)

        x = np.array([[0], [0], [1]])
        y = np.array([[1], [0], [0]])
        z = np.array([[0], [1], [0]])

        R_Grasp2URDF = np.concatenate((x, y, z), axis=1)

        R_target = graspnet_orn @ R_Grasp2URDF

        return R_target

def pq2pbpq(pwxyz):     # pytransform quaternion -> pybullet quaternion
    pwxyz[3], pwxyz[4], pwxyz[5], pwxyz[6] = pwxyz[4], pwxyz[5], pwxyz[6], pwxyz[3]
    return pwxyz

def pbpq2pq(pxyzw):     # pybullet quaternion -> pytransform quaternion
    pxyzw[3], pxyzw[4], pxyzw[5], pxyzw[6] = pxyzw[6], pxyzw[3], pxyzw[4], pxyzw[5]
    return pxyzw






