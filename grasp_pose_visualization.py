import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

#changed
# loading point cloud of the object

obj_number = str(27619)     # change object number for vaious objects
child_link_num = 1


pcd = o3d.io.read_point_cloud(f'./{obj_number}/point_sample/ply-10000.ply')
pcd.scale(0.1, [0, 0, 0])

# loading grasp poses of the object
grasp_data = np.load(f'./{obj_number}/point_sample/grasp_poses.npz', allow_pickle=True)
lst = grasp_data.files
for i in lst:
    print(i)
for i in grasp_data['pred_grasps_cam'].item():
    print(i, grasp_data['pred_grasps_cam'].item()[i].shape)

print(grasp_data['contact_pts'].shape)

# shape of the robot gripper to draw
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

# list of grasp poses to draw
grasp_list = []

# drawing the world coordinate
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

num_draw = 30       # adjust this to select how many grasp poses to visualize
# getting grasp poses of the object part #1 child link - differs for objects
for i in range(grasp_data['pred_grasps_cam'].item()[child_link_num].shape[0]):
    grasp = grasp_data['pred_grasps_cam'].item()[child_link_num][i, :]
    this_score = grasp_data['scores'].item()[child_link_num][i]

    if this_score > 0.5:
        colors = [[0, 0, 1] for j in range(len(lines))]     # blue
    elif this_score > 0.4:
        colors = [[0, 1, 0] for j in range(len(lines))]     # green
    elif this_score > 0.3:
        colors = [[1, 0, 0] for j in range(len(lines))]     # red
    else:
        colors = [[0, 0, 0] for j in range(len(lines))]     # black

    # drawing the robot gripper
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # rotating and translating the robot gripper to grasp poses
    line_set.rotate(grasp[:3, :3], center=[0, 0, 0])
    line_set.translate(grasp[:3, 3])

    #labelling grasp score
    score = str( round( grasp_data['scores'].item()[child_link_num][i], 3) )

    grasp_list.append(line_set)

    # only draw targeted number of  grasp poses
    if i == num_draw:
        break


# visualize the point cloud and grasp poses
o3d.visualization.draw_geometries(grasp_list + [pcd, mesh_frame])
