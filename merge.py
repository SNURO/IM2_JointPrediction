import pybullet as pb
import open3d as o3d
import numpy as np
import logging
import time
import grasp_choosing
import trajectory_gen
import math
import pybullet_data
import os
import sys

from robot import UR5RG2
from arm_motion_UR5_modified import ik_fun, wrap_to_joint_limits, normalized_angle, visualization, imagine

from imm.pybullet_util.typing_extra import Tuple3, TranslationT, QuaternionT
from imm.pybullet_util.bullet_client import BulletClient
from imm.pybullet_util.common import (
    get_joint_limits, get_joint_positions,
    get_link_pose)
import imm.pybullet_util.collision as col

from pytransform3d.transformations import (
    screw_axis_from_screw_parameters, transform_from_pq,
    transform_from_exponential_coordinates, pq_from_transform)
from pytransform3d.rotations import quaternion_from_matrix

MOTION_LIB_PATH = '/home/ro/lab/simulation/motion-planners'
sys.path.append(MOTION_LIB_PATH)
from motion_planners.rrt_connect import birrt

def frame_convert(orn):
    x = np.array([[0],[0],[1]])     # hardcoded Rot (graspnet -> actual URDF/robot ee) 
    y = np.array([[1],[0],[0]])
    z = np.array([[0],[1],[0]])
    
    R_Grasp2URDF = np.concatenate((x,y,z), axis=1)
    
    return orn @ R_Grasp2URDF


def gripper_diff(T):
    z_distance = -0.13      # grasp pose is sampled assuming gripper is Franka-Panda, has to change this to fit RG2
    # -0.15 for 24617

    col1 = np.array([[1], [0], [0], [0]])  # hardcoded gripper difference (graspnet -> actual URDF/robot ee)
    col2 = np.array([[0], [1], [0], [0]])
    col3 = np.array([[0], [0], [1], [0]])
    col4 = np.array([[0], [0], [z_distance], [1]])

    T_Panda2UR5 = np.concatenate((col1, col2, col3, col4), axis=1)

    return T @ T_Panda2UR5


def childpos_parent(bc, parent_id, parent_index, child_id, child_index):        # find T (parent -> child) at grasp pose
    p_world2parent = bc.getLinkState(parent_id, parent_index)[0]        # function used for workaround to attach gripper and child link
    q_world2parent = bc.getLinkState(parent_id, parent_index)[1]
    p_world2child = bc.getLinkState(child_id, child_index)[0]
    q_world2child = bc.getLinkState(child_id, child_index)[1]
    p_parent2world = bc.invertTransform(p_world2parent, q_world2parent)[0]
    q_parent2world = bc.invertTransform(p_world2parent, q_world2parent)[1]
    p_parent2child = bc.multiplyTransforms(p_parent2world, q_parent2world, p_world2child, q_world2child)[0]
    q_parent2child = bc.multiplyTransforms(p_parent2world, q_parent2world, p_world2child, q_world2child)[1]

    return p_parent2child, q_parent2child



def main():
    start_time = time.time()

    object_number = str(148)       # modify this for different object 148(revolute), 27619(prismatic)
    child_index = 1

    file_name = f'./{object_number}/mobility.urdf'
    log_level: str = 'WARN'
    scaler: int = 0.1

    # Configure logging
    logging.root.setLevel(log_level)
    logging.basicConfig()

    # Creating a simulation client
    sim_id = pb.connect(pb.GUI)     # use direct if something runs too slow, no visualization though
    if sim_id < 0:
        raise ValueError('Failed to connect to simulator!')
    bc = BulletClient(sim_id)

    base_pos = (0.7, 0, 0.05)
    base_orn = (0,0,0,1)

    obj_id = bc.loadURDF(file_name, base_pos, base_orn, flags=bc.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                         useFixedBase=True, globalScaling=scaler)
    objloaded_time = time.time()

    non_fixed_joints = []

    # extract all joint informations
    for i in range(bc.getNumJoints(obj_id)):
        info = bc.getJointInfo(obj_id, i)
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

        if jointType != bc.JOINT_FIXED:
            jointstate = bc.getJointState(obj_id, i)
            currentangle = jointstate[0]
            state = bc.getLinkState(obj_id, jointID)  # child link
            linkWorldPosition = state[4]  # position of URDF child frame == joint frame (expressed in world)
            linkWorldOrientation = state[5]  # orientation
            non_fixed_joints.append([jointID, jointAxis,
                                     parentFramePos, parentFrameOrn, linkWorldPosition, linkWorldOrientation,
                                     jointUpperLimit, jointLowerLimit, currentangle, jointType])
    """
        # loading point cloud of the object, visulizes grasp poses
    pcd = o3d.io.read_point_cloud(f'./{object_number}/point_sample/ply-10000.ply')
    pcd.scale(0.1, [0, 0, 0])
    """

        # loading grasp poses of the object
    grasp_data = np.load(f'./{object_number}/point_sample/grasp_poses.npz', allow_pickle=True)
    lst = grasp_data.files


        # load joint information
    artobj = bc.loadURDF(f'./{object_number}/mobility.urdf', flags=bc.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                            useFixedBase=True, globalScaling=0.1)
        # 699 grasps for link0, 101 grasps for link1

    """
    for i in grasp_data['pred_grasps_cam'].item():
        print(i, grasp_data['pred_grasps_cam'].item()[i].shape) #is (699,4,4) (101,4,4)
        print(i, grasp_data['scores'].item()[i].shape)  #is  (699,) (101,)
    """

    joint_type, w, q = grasp_choosing.joint_info_ext(artobj)

    # all SE(3) grasp poses for child link, Num_grasp X 4 X 4
    grasp_candidates = grasp_data['pred_grasps_cam'].item()[child_index]
    grasp_scores = grasp_data['scores'].item()[child_index]  # corresponding scores, vector of Num_grasp

        # takes top from_topk grasps according to grasp scores, and choose best operability score among them
    from_topk = 20  # can be modifed, chooses one with highest operability score among k top grasp score grasps

    if joint_type:  # prismatic
        best_index = grasp_choosing.grasp_chooser(grasp_candidates, grasp_scores, 1, w, q)
    else:  # revolute
        best_index = grasp_choosing.grasp_chooser(grasp_candidates, grasp_scores, from_topk, w, q)

    ##grasp_choosing.visualizer([best_index], grasp_candidates, grasp_scores, pcd)

    bc.removeBody(artobj)

    # convert grasp pose to fit (Panda-Franka -> RG2)
    grasp_T = gripper_diff(grasp_candidates[best_index])
    grasp_pos = grasp_T[:3,3]
    grasp_orn = quaternion_from_matrix(frame_convert(grasp_T[:3,:3]))
    graspselect_time = time.time()

    grasp_orn[0], grasp_orn[1], grasp_orn[2], grasp_orn[3] = grasp_orn[1], grasp_orn[2], grasp_orn[3], grasp_orn[0]
    # to fit pybullet quaternion, xyzw

    traj_plan = trajectory_gen.trajectory(bc, obj_id, base_pos,
                                          base_orn, grasp_pos, grasp_orn, non_fixed_joints, currentangle)
    traj_plan.visualization(traj_plan.gripper_trajectories_down)
    traj_plan.visualization(traj_plan.gripper_trajectories_up)

    bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING, 1)

    trajplan_time = time.time()

######################################################################### Robot arm manipulation from here

    seed: int = 0
    max_pos_tol: float = 0.05
    max_orn_tol: float = np.deg2rad(5)
    num_ik_iter: int = 1024
    max_ik_residual: float = 0.01
    delay: float = 1/240
    log_level: str = 'DEBUG'
    line_color: Tuple3[float] = (0, 0, 1)
    line_width: float = 4
    line_lifetime: float = 50.0

    # Configure logging.
    logging.root.setLevel(log_level)
    logging.basicConfig()

    # Load scene.
    bc.setAdditionalSearchPath(
        pybullet_data.getDataPath())
    plane_id: int = bc.loadURDF('plane.urdf', (0, 0, -0.3))
    robot = UR5RG2(bc, (0, 0, 0), (0, 0, 0, 1))
    robot.reset_arm_poses()
    robot_id = robot.robot_id
    ee_id = robot.ee_id
    joint_ids = robot.arm_joint_ids
    joint_limits = robot.arm_joint_limits
    robotloaded_time = time.time()

    # Current pose
    bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING, 1)
    q_src = get_joint_positions(bc, robot_id, joint_ids)
    src_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)


    # target EE pose.
    target_pos = traj_plan.gripper_pos
    target_orn = traj_plan.gripper_orn   # xyzw

    # Run inverse kinematics.
    q_dst = ik_fun(bc, robot_id, ee_id, target_pos, target_orn,
                   maxNumIterations=num_ik_iter,
                   residualThreshold=max_ik_residual,
                   )[:6]

    # Ensure within joint limits.
    q_dst = wrap_to_joint_limits(q_dst, joint_limits)
    if (np.any(q_dst < joint_limits[0])
            or np.any(q_dst >= joint_limits[1])):
        logging.debug('skip due to joint limit violation')
    for i, v in zip(joint_ids, q_dst):
        bc.resetJointState(robot_id, i, v)

    # Check EE pose at specified joint positions.
    # NOTE(ycho): calculateInverseKinematics() operates on
    # the link coordinate, not CoM(inertial) coordinate.
    dst_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)

    d_pos = np.subtract(target_pos, dst_ee_pose[0])
    d_ang = bc.getAxisAngleFromQuaternion(
        bc.getDifferenceQuaternion(target_orn, dst_ee_pose[1]))
    print(f'pos deviation : {np.linalg.norm(d_pos)}, ang deviation : {normalized_angle(d_ang[1])}')
    if np.linalg.norm(d_pos) >= max_pos_tol:
        logging.debug('skip due to positional deviation')
    if abs(normalized_angle(d_ang[1])) >= max_orn_tol:
        logging.debug('skip due to orientation deviation')

    bc.performCollisionDetection() #
    if len(bc.getContactPoints(robot_id)) > 0:
        logging.debug(f'skip due to collision: Contact point is {bc.getContactPoints(robot_id)}')
    place_atstart_time = time.time()

    # workaround - attaching gripper with child link (removed)
    """childpos_fromparent = childpos_parent(bc, robot_id, ee_id, obj_id, child_index) 
    
    bc.createConstraint(parentBodyUniqueId = robot_id, parentLinkIndex = ee_id, childBodyUniqueId = obj_id,
                        childLinkIndex = child_index, jointType = bc.JOINT_FIXED, jointAxis = [0,0,0],
                        parentFramePosition = childpos_fromparent[0], childFramePosition = [0,0,0],
                        parentFrameOrientation = childpos_fromparent[1], childFrameOrientation = [0,0,0,1])"""


    robot.close_gripper()
    for _ in range(30):
        bc.stepSimulation()
        time.sleep(delay)
    print('gripper_closed')

    dst_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)

    # print deviation from targeted pose and actual result
    d_pos = np.subtract(target_pos, dst_ee_pose[0])
    d_ang = bc.getAxisAngleFromQuaternion(
        bc.getDifferenceQuaternion(target_orn, dst_ee_pose[1]))
    print(f'pos deviation : {np.linalg.norm(d_pos)}, ang deviation : {normalized_angle(d_ang[1])}')
    if np.linalg.norm(d_pos) >= max_pos_tol:
        logging.debug('skip due to positional deviation')
    if abs(normalized_angle(d_ang[1])) >= max_orn_tol:
        logging.debug('skip due to orientation deviation')

    # visualizing actual result SE(3)
    vis_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)
    dis_ee_pose = list(vis_pose[0] + vis_pose[1])
    visualization(bc, dis_ee_pose)
    
    grip_Collision_allowed = []     # allow contact between finger and object child link. allowlist -> collision_fn -> birrt

    # right finger, left finger
    linkpair_1 = col.LinkPair(body_id_a = robot_id, link_id_a = 11, body_id_b = obj_id, link_id_b = child_index)
    grip_Collision_allowed.append(linkpair_1)
    linkpair_2 = col.LinkPair(body_id_a=robot_id, link_id_a=14, body_id_b=obj_id, link_id_b=child_index)
    grip_Collision_allowed.append(linkpair_2)

    bc.performCollisionDetection()           # this is also used in collision_fn in birrt, to test
    contacts = bc.getContactPoints(bodyA=robot_id)

    # collision detected, but ignored in birrt
    #print(contacts)

    # Compute motion plan.
    collision_fn = col.ContactBasedCollision(bc,
                                         robot_id, joint_ids,
                                         grip_Collision_allowed, [], joint_limits, {})


    def distance_fn(q0: np.ndarray, q1: np.ndarray):
        return np.linalg.norm(np.subtract(q1, q0))

    def sample_fn():
        return rng.uniform(joint_limits[0], joint_limits[1])

    def extend_fn(q0: np.ndarray, q1: np.ndarray):
        dq = np.subtract(q1, q0)  # Nx6
        return q0 + np.linspace(0, 1)[:, None] * dq

    timestamp = 0
    action_seq = []

    for i in range(len(traj_plan.gripper_trajectories_up)):

        q_src = get_joint_positions(bc, robot_id, joint_ids)

        target_pos = traj_plan.gripper_trajectories_up[i][:3]
        target_orn = trajectory_gen.pq2pbpq(traj_plan.gripper_trajectories_up[i])[3:]  # xyzw

        # Run inverse kinematics.
        q_dst = ik_fun(bc, robot_id, ee_id, target_pos, target_orn,
                       maxNumIterations=num_ik_iter,
                       residualThreshold=max_ik_residual,
                       )[:6]

        with imagine(bc):
            q_trajectory = birrt(
                q_src,
                q_dst,
                distance_fn,
                sample_fn,
                extend_fn,
                collision_fn)

        # Disable default joints.
        bc.setJointMotorControlArray(
            robot_id,
            joint_ids,
            bc.VELOCITY_CONTROL,
            targetVelocities=np.zeros(len(joint_ids)),
            forces=np.zeros(
                len(joint_ids)))

        # Execute the trajectory.
        prv_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)

        for q in q_trajectory:
            bc.setJointMotorControlArray(robot_id, joint_ids,
                                         bc.POSITION_CONTROL,
                                         targetPositions=q)
            bc.stepSimulation()
            timestamp += 1
            # below line is to get observation sequence, slows down in GUI, works fine with pb.DIRECT
            #width, height, rgb, depth, seg = bc.getCameraImage(320, 200, renderer=bc.ER_TINY_RENDERER)
            cur_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)
            action_seq.append(cur_ee_pose)      # save current pose

            bc.addUserDebugLine(
                prv_ee_pose[0],
                cur_ee_pose[0],
                line_color,
                line_width, line_lifetime)
            prv_ee_pose = cur_ee_pose
            time.sleep(delay)

    end_time = time.time()

    print('done')

    print(f'obj loaded time : {round(objloaded_time - start_time, 3)}s \n '
          f'grasp selection time : {round(graspselect_time - objloaded_time, 3)}s \n '
          f'traj plan time : {round(trajplan_time - graspselect_time, 3)}s \n '
          f'robot loaded time : {round(robotloaded_time - trajplan_time, 3)}s \n '
          f'place at start time : {round(place_atstart_time - robotloaded_time, 3)}s \n '
          f'motion exec time : {round(end_time - place_atstart_time, 3)}s \n ')



if __name__ == '__main__':
    main()
