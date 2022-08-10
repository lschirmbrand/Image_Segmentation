import copy
import math
import sys
import numpy as np
import struct
import open3d as o3d
from camera import Camera
from quaternion import *

from scipy.spatial.transform import Rotation

pcd_reference = o3d.io.read_point_cloud("./bin_to_ply/sparse_reference.ply")
pcd_testobject = o3d.io.read_point_cloud("./bin_to_ply/sparse_ref_test.ply")

_, ind1 = pcd_reference.remove_radius_outlier(nb_points=15, radius=0.15)
pcd_reference = pcd_reference.select_by_index(ind1)

_, ind2 = pcd_testobject.remove_radius_outlier(nb_points=16, radius=0.1)
pcd_testobject = pcd_testobject.select_by_index(ind2)

bb_pcd_reference = pcd_reference.get_oriented_bounding_box()

o = np.zeros((1,3))
# print(pcd_reference)
# print(pcd_oversized)


# print(o)
color = np.zeros((1, 3))
color[0][0] = 255
color[0][1] = 0
color[0][2] = 0
pcd_coordinate_system = o3d.geometry.PointCloud()
pcd_coordinate_system.points = o3d.utility.Vector3dVector(o)
pcd_coordinate_system.colors = o3d.utility.Vector3dVector(color.astype(float)/255)

# print(pcd_coordinate_system)

points = [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]]
lines = [[0, 1], [0, 2], [0, 3]]
# colors = [[1, 0, 0] for i in range(len(lines))]
colors = [[1, 0, 0], [0,1,0], [0,0,1]]

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)


def translate(pcd, translation):
    return pcd.translate(translation)
    

def init_cameras(filepath):
    cameras = []

    with open(filepath) as f:
        lines = f.readlines()

        for l in lines:
            if l.startswith('#'):
                continue
            qw, qx, qy, qz, tx, ty, tz = l.split(" ")
            qx, qy, qz, qw = normalize([float(qx), float(qy), float(qz), float(qw)])
            cam = Camera(float(qw), -float(qx), -float(qy), -float(qz), float(tx), float(ty), float(tz))
            # print(cam.qw, cam.qx, cam.qy, cam.qz)
            cameras.append(cam)
    f.close()
    return cameras

def calculate_camera_position(camera):
    P = np.array([camera.tx, camera.ty, camera.tz])
    R = get_camera_rot(camera)
    # print(P)
    # print(R)
    P = np.dot(P, R.T)
    return P

def get_all_camera_positions(cameras):
    cam_points = np.zeros((len(cameras), 3))
    for i, camera in enumerate(cameras):
        cam_points[i] = calculate_camera_position(camera)
    return cam_points

def get_camera_rot(camera):
    return Rotation.from_quat([camera.qx, camera.qy, camera.qz, camera.qw]).as_matrix()

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def generate_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=ransac_n,
                                         num_iterations=num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    return inlier_cloud, plane_model

def get_rotation_matrix(plane_model, rotation_axis):
    [a, b, c, d] = plane_model
    pcd_normal_vector = np.array([a, b, c])
    
    R = rotate_to_normal(pcd_normal_vector, rotation_axis)
    return R


def unitcross(a, b):
    return np.cross(a, b) / np.linalg.norm(np.cross(a, b))

def rotate_to_normal(pcd_normal, plane_normal):
    costheta = np.dot(pcd_normal,plane_normal)/(np.linalg.norm(pcd_normal)*np.linalg.norm(plane_normal))
    axis = unitcross(pcd_normal, plane_normal)

    c = costheta
    C = 1 - c
    s = np.sqrt(1 - c**2)

    # print(axis)
    x = axis[0]
    y = axis[1]
    z = axis[2]
    # print(y)

    R = np.matrix([[x**2*C + c, x*y*C - z*s, x*z*C + y*s], 
                    [y*x*C + z*s, y**2*C + c, y*z*C - x*s], 
                    [z*x*C - y*s, z*y*C + x*s, z*z*C + c]])
    return R

def rotate_to_center_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])

    center = inlier_cloud.get_center()
    # print("Center:", center)
    pcd.translate((-center[0],-center[1],-center[2]), relative=True)

    pcd_normal_vector = np.array([a, b, c])

    R = rotate_to_normal(pcd_normal_vector, [1,0,0])
    pcd.rotate(R, center=(0,0,0))

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                          ransac_n=3,
                                          num_iterations=1000)


    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    return pcd, inlier_cloud

def apply_rotation(pcd, R, center):
    pcd.rotate(R, center=center)
    return pcd


#  REFFERENCE OBJECT

center = pcd_reference.get_center()
# print(center)

cameras_ref = init_cameras('bin_to_ply/cameras_ref.txt')
cameras_ref_points = get_all_camera_positions(cameras_ref)

camera_ref_pcd = o3d.geometry.PointCloud()
camera_ref_pcd.points = o3d.utility.Vector3dVector(cameras_ref_points)

camera_ref_plane, plane_model = generate_plane(camera_ref_pcd, distance_threshold=100, ransac_n=len(cameras_ref))
camera_plane_ref_R = get_rotation_matrix(plane_model, [0,1,0])

[a,b,c,d] = plane_model
print(f"Plane equation cam: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

#  TEST OBJECT
cameras_object = init_cameras('bin_to_ply/cameras_test.txt')
cameras_object_points = get_all_camera_positions(cameras_object)

camera_test_pcd = o3d.geometry.PointCloud()
camera_test_pcd.points = o3d.utility.Vector3dVector(cameras_object_points)

camera_ref_plane, plane_model = generate_plane(camera_test_pcd, distance_threshold=100, ransac_n=len(cameras_object))
camera_plane_ref_R = get_rotation_matrix(plane_model, [0,1,0])

[a,b,c,d] = plane_model
print(f"Plane equation cam: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

pcd_ref_plane , plane_model = generate_plane(pcd_reference)
pcd_reference_R = get_rotation_matrix(plane_model, [1,0,0])

[a,b,c,d] = plane_model
print(f"Plane equation ref: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

R = camera_plane_ref_R @ pcd_reference_R

# pcd_ref_rotated_2 = apply_rotation(copy.deepcopy(pcd_reference), pcd_reference_R, (0,0,0))
# pcd_ref_rotated_3 = apply_rotation(copy.deepcopy(pcd_reference), pcd_reference_R, (0,0,0))
# pcd_ref_rotated_3 = apply_rotation(copy.deepcopy(pcd_reference), camera_plane_R, (0,0,0))
pcd_ref_rotated = apply_rotation(copy.deepcopy(pcd_reference), R, (0,0,0))
bb_pcd_reference = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_ref_rotated.points)



# o3d.visualization.draw_geometries([bb_ref, bb_oversized, ref_in, over_in, pcd_coordinate_system, pcd_reference, pcd_oversized, line_set])
o3d.visualization.draw_geometries([line_set, pcd_reference, camera_ref_pcd, camera_test_pcd, pcd_testobject])
# o3d.visualization.draw_geometries([ref_in, bb_ref, pcd_coordinate_system, pcd_reference, line_set])
# o3d.visualization.draw_geometries([bb, pcd_coordinate_system, pcd_reference, line_set])
# o3d.visualization.draw_geometries([pcd_coordinate_system, pcd_reference, pcd_oversized, line_set])
# o3d.visualization.draw_geometries([ref_in, pcd_coordinate_system, pcd_reference, line_set])
# o3d.visualization.draw_geometries([bb_pcd_reference, pcd_coordinate_system, pcd_reference_old, pcd_reference, line_set, camera_pcd])
# o3d.visualization.draw_geometries([pcd_coordinate_system, line_set, camera_plane, pcd_ref_rotated, pcd_ref_plane])

