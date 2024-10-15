import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

import yaml
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import  getWorld2View2, focal2fov, fov2focal
# from utils.dataset_utils import SceneInfo, CameraInfo, getNerfppNorm, storePly, fetchPly
from scipy.spatial.transform import Rotation as R
from scene.gaussian_model import BasicPointCloud
import natsort
## Read camera information
# CameraInfo(
#     uid=idx,
#     R=R,
#     T=T,
#     FovY=fovy,
#     FovX=fovx,
#     image_path=image_path,
#     image_name=image_name,
#     width=width,
#     height=height,
#     intrinsics=intrinsics,
# )
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def read_camera_params(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    intrinsics = {
        "fx": data['Camera1.fx'],
        "fy": data['Camera1.fy'],
        "cx": data['Camera1.cx'],
        "cy": data['Camera1.cy'],
        "k1": data['Camera1.k1'],
        "k2": data['Camera1.k2'],
        "k3": data['Camera1.k3'],
        "p1": data['Camera1.p1'],
        "p2": data['Camera1.p2']
    }
    K = np.array([
    [intrinsics['fx'], 0, intrinsics['cx']],
    [0, intrinsics['fy'], intrinsics['cy']],
    [0, 0, 1]
    ])
    width = data['Camera.width']
    height = data['Camera.height']
    fovx = focal2fov(intrinsics['fx'], width)
    fovy = focal2fov(intrinsics['fy'], height)
    print("FOV X", fovx)
    print("FOV Y", fovy)
    
    return K, fovx, fovy, width, height

def read_camera_trajectory(trajectory_file):
    poses = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            timestamp = parts[0]
            translation = np.array(parts[1:4])
            rotation = np.array(parts[4:])
            Rt  =  R.from_quat(rotation).as_matrix().T
            poses.append((timestamp, translation, Rt))
    return poses

# def read_camera_trajectory_from_json(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)
    
#     poses = []
    
#     for entry in data:
#         R = np.array(entry["R"] )
#         T = np.array(entry["t"])
#         poses.append((entry["id"], T, R))
    
#     return poses
def read_camera_trajectory_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    poses = []
    
    for entry in data:
        transform_matrix = np.array([
        [-1, 0,  0],
        [0, 1, 0],  # Invert Y-axis
        [0, 0, 1]   # Invert Z-axis
        ]   )
        translation = np.array(entry["position"])
        rotation = np.array(entry["rotation"])


        Rt = np.eye(4)
        Rt[:3, :3] = rotation
        Rt[:3, 3] = translation

        Rt[:3, 0] *= -1  # Negate x-axis

        C2W = np.linalg.inv(Rt)

        camera_R = transform_matrix @ C2W[:3, :3]
        camera_T = C2W[:3, 3]

        poses.append((entry["id"], camera_T, camera_R))

        # poses.append((entry["id"], T_colmap, R))
    
    return poses

def get_image_list(image_folder):
    rgb_images = natsort.natsorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
    depth_images = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if "depth" in f])
    return rgb_images, depth_images

def readPhotoSlamInfo(path, yaml_file, trajectory_file, image_folder):
    cam_infos = []
    
    intrinsics, fovx, fovy, width, height = read_camera_params(yaml_file)
    poses =read_camera_trajectory_from_json(trajectory_file)
    rgb_images, depth_images = get_image_list(image_folder)
    
    for idx, (uid, translation, rotation) in enumerate(poses):
        R = rotation
        T = translation
        image_path = rgb_images[idx]  # Assume corresponding RGB and depth images
        # image_name = os.path.basename(image_path)
        image_name_with_ext = os.path.basename(image_path)
        image_name = os.path.splitext(image_name_with_ext)[0]
        image = Image.open(image_path)
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            # intrinsics=intrinsics
        )
        cam_infos.append(cam_info)
    nerf_normalization = getNerfppNorm(cam_infos)
    ply_path = os.path.join(path, "ply/input.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)
    
    eval = False
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info