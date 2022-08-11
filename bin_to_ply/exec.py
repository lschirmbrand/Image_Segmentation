import copy
import math
import sys
from tracemalloc import start
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

lineset = o3d.geometry.LineSet()
lineset.points = o3d.utility.Vector3dVector(points)
lineset.lines = o3d.utility.Vector2iVector(lines)
lineset.colors = o3d.utility.Vector3dVector(colors)


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

    print(axis)

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

def apply_rotation_pcd(pcd, R, center):
    pcd.rotate(R, center=center)
    return pcd

def calculate_center(points3d):
    return np.mean(points3d, axis=0)

def get_max_bounds(points3d, variance=0.01):
    for p in points3d:
        for i, p2 in enumerate(points3d):
            if p is p2:
                continue
            else:
                if(abs(p[0] - p2[0]) < variance and abs(p[1] - p2[1]) < variance and abs(p[2] - p2[2]) < variance):
                    p[0] = p2[0]
                    p[1] = p2[1]
                    p[2] = p2[2]

    points3d = np.unique(points3d, axis=0)
    
    if(len(points3d) != 4):
        print("Error: not enough points")
        return None
    else: return points3d


def get_outer_cameras(cameras, points3d):
    outer_cameras = []
    for point in points3d:
        distances = []

        for i, cam in enumerate(cameras):
            distances.append(calculate_distance(point, cam))
        
        min_dis = min(distances)
        min_index = distances.index(min_dis)
        outer_cameras.append(cameras[min_index])
        # cameras.remove(cameras[min_index])
    return outer_cameras
    

def calculate_distance(p1, p2):
    return np.linalg.norm(p1-p2)

def get_smallest_distance(points, p):
    distances = []
    for point in points:
        distances.append(calculate_distance(p, point))
    return min(distances), distances.index(min(distances))

def get_highest_distance(points, p):
    distances = []
    for point in points:
        distances.append(calculate_distance(p, point))
    return max(distances), distances.index(max(distances))

def calculate_vector(p1, p2):
    return p2-p1

def apply_translation_to_points(translation, points):
    for i, point in enumerate(points):
        points[i] = point + translation
    return points

def apply_rotation_to_points(P, R):
    return np.asarray(np.dot(P, R.T))


def rearrange_cloud():
    pass

#  REFFERENCE OBJECT

center = pcd_reference.get_center()
# print(center)

cameras_ref = init_cameras('bin_to_ply/cameras_ref.txt')
cameras_ref_points = get_all_camera_positions(cameras_ref)

cam_ref_center = calculate_center(cameras_ref_points)
# print(cam_ref_center)

camera_ref_pcd = o3d.geometry.PointCloud()
camera_ref_pcd.points = o3d.utility.Vector3dVector(cameras_ref_points)

camera_ref_pcd = translate(camera_ref_pcd, -cam_ref_center)

cam_ref_center = camera_ref_pcd.get_center()

cam_ref_center_pcd = o3d.geometry.PointCloud()
cam_ref_center_pcd.points = o3d.utility.Vector3dVector([cam_ref_center])

cam_ref_pcd_bb = camera_ref_pcd.get_oriented_bounding_box()
corner_points = np.asarray(cam_ref_pcd_bb.get_box_points())

corner_points = get_max_bounds(corner_points, variance=0.2)
outer_cams = get_outer_cameras(np.asarray(camera_ref_pcd.points), corner_points)

print(outer_cams)

_, index_start = get_smallest_distance(outer_cams, (0,0,0))
print(index_start)
start_cam = outer_cams[index_start]
outer_cams = np.delete(outer_cams, index_start, axis=0)

_, index_remove = get_highest_distance(outer_cams, start_cam)
outer_cams = np.delete(outer_cams, index_remove, axis=0)

cam_vector_points = np.concatenate((start_cam.reshape(1,3), outer_cams), axis=0)

t = calculate_vector(cam_vector_points[0], (0,0,0))
cam_vector_points = apply_translation_to_points(t, cam_vector_points)
camera_ref_pcd.translate(t, relative=True)

print(cam_vector_points)
R1 = rotate_to_normal(calculate_vector(cam_vector_points[0], cam_vector_points[1]), [1,0,0])
cam_vector_points = apply_rotation_to_points(cam_vector_points, R1)

vec = calculate_vector(cam_vector_points[0], cam_vector_points[2])
vec[0] = 0
# print("vec ", vec)
print(cam_vector_points)
R2 = rotate_to_normal(vec, [0,0,1])

cam_vector_points = apply_rotation_to_points(cam_vector_points, R2)

camera_ref_pcd.rotate(R1, center=(0,0,0))
camera_ref_pcd.rotate(R2, center=(0,0,0))

pcd_reference.translate(t, relative=True)
pcd_reference.rotate(R1, center=(0,0,0))
pcd_reference.rotate(R2, center=(0,0,0))

print("cam_vector_points \n", cam_vector_points)

cam_vectors = o3d.geometry.LineSet()
cam_vectors.points = o3d.utility.Vector3dVector(cam_vector_points)
cam_vectors.lines = o3d.utility.Vector2iVector([[0,1], [0,2], [1,2]])
cam_vectors.colors = o3d.utility.Vector3dVector([[1, 0, 0]])


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
pcd_ref_rotated = apply_rotation_pcd(copy.deepcopy(pcd_reference), R, (0,0,0))
bb_pcd_reference = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_ref_rotated.points)



# o3d.visualization.draw_geometries([bb_ref, bb_oversized, ref_in, over_in, pcd_coordinate_system, pcd_reference, pcd_oversized, line_set])
o3d.visualization.draw_geometries([cam_vectors, camera_ref_pcd, cam_ref_center_pcd, cam_vectors, lineset, pcd_reference])
# o3d.visualization.draw_geometries([ref_in, bb_ref, pcd_coordinate_system, pcd_reference, line_set])
# o3d.visualization.draw_geometries([bb, pcd_coordinate_system, pcd_reference, line_set])
# o3d.visualization.draw_geometries([pcd_coordinate_system, pcd_reference, pcd_oversized, line_set])
# o3d.visualization.draw_geometries([ref_in, pcd_coordinate_system, pcd_reference, line_set])
# o3d.visualization.draw_geometries([bb_pcd_reference, pcd_coordinate_system, pcd_reference_old, pcd_reference, line_set, camera_pcd])
# o3d.visualization.draw_geometries([pcd_coordinate_system, line_set, camera_plane, pcd_ref_rotated, pcd_ref_plane])

