import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import pybullet as pb

from pytransform3d.transformations import (
    screw_axis_from_screw_parameters, transform_from_pq,
    transform_from_exponential_coordinates, pq_from_transform)


def joint_info_ext(object):
    # List for saving only movable joints
    non_fixed_joints = []

    for i in range(pb.getNumJoints(object)):
        info = pb.getJointInfo(object, i)
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

        if jointType != pb.JOINT_FIXED:
            state = pb.getLinkState(object, jointID)
            linkWorldPosition = state[0]
            linkWorldOrientation = state[1]
            non_fixed_joints.append([jointID, jointAxis,
                                     parentFramePos, parentFrameOrn, linkWorldPosition, linkWorldOrientation,
                                     jointUpperLimit, jointLowerLimit, jointType])

    for j in non_fixed_joints:
        pq_world2center = np.concatenate(([0, 0, 0], linkWorldOrientation))  # COM not defined in URDF. modify if it is
        T_world2center = transform_from_pq(pq_world2center)

        s_axis = np.matmul(T_world2center, np.concatenate((jointAxis, np.zeros(1))))[:3]
        q = np.array(linkWorldPosition)

        return non_fixed_joints[0][8], s_axis, q  # type


# indices : list, index of grasps that are wanted to be visualized
# grasps : list of SE(3) grasps, num X 4 X 4
def visualizer(indices, grasps, scores, pcd):
    points = [
        [0, 0, 0],
        [0, 0, 0.0624],
        [0.0399, 0, 0.0624],
        [-0.0399, 0, 0.0624],
        [0.0399, 0, 0.1034],
        [-0.0399, 0, 0.1034]
    ]
    lines = [
        [0, 1],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5]
    ]

    grasp_list = []

    # drawing the world coordinate
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    num_draw = len(indices)

    # getting grasp poses of the object part #1 (handle)
    for i in range(num_draw):
        this_grasp = grasps[indices[i], :, :]
        this_score = scores[indices[i]]

        if this_score > 0.5:
            colors = [[0, 0, 1] for j in range(len(lines))]
        elif this_score > 0.4:
            colors = [[0, 1, 0] for j in range(len(lines))]
        elif this_score > 0.3:
            colors = [[1, 0, 0] for j in range(len(lines))]
        else:
            colors = [[0, 0, 0] for j in range(len(lines))]

        # drawing the robot gripper
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                        lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # rotating and translating the robot gripper to grasp poses
        line_set.rotate(this_grasp[:3, :3], center=[0, 0, 0])
        line_set.translate(this_grasp[:3, 3])

        grasp_list.append(line_set)

    # visualize the point cloud and grasp poses
    o3d.visualization.draw_geometries(grasp_list + [pcd, mesh_frame])


#   visualize all grasps, green has high scores, blue has low scores
def visualizer_all(indices, grasps, scores):
    points = [
        [0, 0, 0],
        [0, 0, 0.0624],
        [0.0399, 0, 0.0624],
        [-0.0399, 0, 0.0624],
        [0.0399, 0, 0.1034],
        [-0.0399, 0, 0.1034]
    ]
    lines = [
        [0, 1],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5]
    ]

    grasp_list = []

    # drawing the world coordinate
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    num_draw = len(indices)

    # getting grasp poses of the object part #1 (handle)
    for i in range(num_draw):
        this_grasp = grasps[indices[i], :, :]
        this_score = scores[indices[i]]

        # Black, Red, Green, Blue : getting better grasp scores
        if this_score > 0.5:
            colors = [[0, 0, 1] for j in range(len(lines))]
        elif this_score > 0.4:
            colors = [[0, 1, 0] for j in range(len(lines))]
        elif this_score > 0.3:
            colors = [[1, 0, 0] for j in range(len(lines))]
        else:
            colors = [[0, 0, 0] for j in range(len(lines))]

        # drawing the robot gripper
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                        lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # rotating and translating the robot gripper to grasp poses
        line_set.rotate(this_grasp[:3, :3], center=[0, 0, 0])
        line_set.translate(this_grasp[:3, 3])

        grasp_list.append(line_set)

    # visualize the point cloud and grasp poses
    o3d.visualization.draw_geometries(grasp_list + [pcd, mesh_frame])


def grasp_chooser(candidates, scores, num_grasp, w, q):
    def find_distance(grasp_origin, screw_w, screw_q):  # finding distance between gripper finger center and joint axis
        distance = np.cross(grasp_origin - screw_q, screw_w)
        distance = np.linalg.norm(distance)
        distance = distance / np.linalg.norm(screw_w)

        return distance

    tmpscores = scores.copy()
    index = []
    for i in range(num_grasp):
        tmpindex = tmpscores.argmax()
        index.append(tmpindex)
        tmpscores[tmpindex] = 0

    current_max_distance = 0
    current_max_index = -1

    for j in index:
        T_world2grasp = candidates[j, :, :]
        G_grasp = np.array([0, 0, 0.1034, 1])       # position of gripper center seen from end-effector frame
        G_world = (T_world2grasp @ G_grasp)[:3]     # position of gripper center seen from world frame

        d = find_distance(G_world, w, q)
        if d > current_max_distance:
            current_max_index = j
            current_max_distance = d

    print('current_max_distance')
    print(current_max_distance)

    return current_max_index


def grasp_chooser_maxgraspscore(candidates, scores, num_grasp):
    tmpscores = scores.copy()
    index = []
    for i in range(num_grasp):
        tmpindex = tmpscores.argmax()
        index.append(tmpindex)
        tmpscores[tmpindex] = 0

    return index


if __name__ == '__main__':

    object_num = str(148)
    child_num = 1
    # changed
    # loading point cloud of the object
    pcd = o3d.io.read_point_cloud(f'./{object_num}/point_sample/ply-10000.ply')
    pcd.scale(0.1, [0, 0, 0])

    # loading grasp poses of the object
    grasp_data = np.load(f'./{object_num}/point_sample/grasp_poses.npz', allow_pickle=True)
    lst = grasp_data.files

    # load joint information
    physicsClient = pb.connect(pb.DIRECT)
    artobj = pb.loadURDF(f'./{object_num}/mobility.urdf', flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                         useFixedBase=True, globalScaling=0.1)
    # 699 grasps for link0, 101 grasps for link1

    """
    #for i in lst:
        #print(i)
    for i in grasp_data['pred_grasps_cam'].item():
        print(i, grasp_data['pred_grasps_cam'].item()[i].shape) #is (699,4,4) (101,4,4)
        print(i, grasp_data['scores'].item()[i].shape)  #is  (699,) (101,)
    """

    joint_type, w, q = joint_info_ext(artobj)

    # all SE(3) grasp poses for child link, Num_grasp X 4 X 4
    grasp_candidates = grasp_data['pred_grasps_cam'].item()[child_num]
    grasp_scores = grasp_data['scores'].item()[child_num]  # corresponding scores, vector of Num_grasp

    # takes top from_topk grasps according to grasp scores, and choose best operability score among them
    from_topk = 20  # may have to be modified

    if joint_type:  # prismatic, distance doesn't matter significantly in this case
        best_index = grasp_chooser(grasp_candidates, grasp_scores, 5, w, q)
    else:  # revolute
        best_index = grasp_chooser(grasp_candidates, grasp_scores, from_topk, w, q)

    print(best_index)
    print(grasp_candidates[best_index])
    visualizer([best_index], grasp_candidates, grasp_scores, pcd)


    """
    top_grasp_score_indices = grasp_chooser_maxgraspscore(grasp_candidates, grasp_scores, 50)
    visualizer(top_grasp_score_indices, grasp_candidates, grasp_scores)
    """