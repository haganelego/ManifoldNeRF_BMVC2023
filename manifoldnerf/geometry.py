import numpy as np
from scipy.spatial.transform import Rotation
import torch

rot_x = lambda phi : np.array([
    [1,0,0,],
    [0,np.cos(phi),-np.sin(phi)],
    [0,np.sin(phi), np.cos(phi)]], dtype=np.float64)

rot_y = lambda th : np.array([
    [np.cos(th),0,np.sin(th)],
    [0,1,0],
    [-np.sin(th),0, np.cos(th)]], dtype=np.float64)

rot_z = lambda th : np.array([
    [np.cos(th), -np.sin(th) ,0],
    [np.sin(th), np.cos(th) ,0],
    [0 ,0, 1]], dtype=np.float64)

def cartesian2polar(position):
    newPosition = np.empty([3], dtype=np.float64)
    newPosition[0] = np.linalg.norm(position, axis=0)
    newPosition[1] = np.arccos(position[2] / newPosition[0])
    newPosition[2] = np.arctan2(position[1], position[0])
    nan_index = np.isnan(newPosition[1])
    newPosition[nan_index,1] = 0
    return newPosition


def slerp(p0, p1, t):
    # https://stackoverflow.com/questions/2879441/how-to-interpolate-rotations
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def interp(pose1, pose2, s):
    """Interpolate between poses as camera-to-world transformation matrices"""
    assert pose1.shape == (3, 4)
    assert pose2.shape == (3, 4)

    # Camera translation 
    C = (1 - s) * pose1[:, -1] + s * pose2[:, -1]
    assert C.shape == (3,)

    # Rotation from camera frame to world frame
    R1 = Rotation.from_matrix(pose1[:, :3])
    R2 = Rotation.from_matrix(pose2[:, :3])
    R = slerp(R1.as_quat(), R2.as_quat(), s)
    R = Rotation.from_quat(R)
    R = R.as_matrix()
    assert R.shape == (3, 3)
    transform = np.concatenate([R, C[:, None]], axis=-1)
    return torch.tensor(transform, dtype=pose1.dtype)

def interp_rot(theta, phi):
    c2w = rot_z(phi + np.pi / 2.0)
    c2w = c2w @ rot_x(theta)
    return c2w

def interp_pose(pose1, pose2, s, center=None):
    """Interpolate between poses as camera-to-world transformation matrices"""
    assert pose1.shape == (3, 4)
    assert pose2.shape == (3, 4)

    # Camera translation 
    # C = (1 - s) * pose1[:, -1] + s * pose2[:, -1]
    if center is not None:
        center_np = np.array(center)
        C = slerp(pose1[:, -1] - center_np, pose2[:, -1] - center_np, s)
        C = C + np.array(center)
    else:
        C = slerp(pose1[:, -1], pose2[:, -1], s)
    assert C.shape == (3,)

    # Rotation from camera frame to world frame
    polar_pose = cartesian2polar(C)
    R = interp_rot(polar_pose[1], polar_pose[2])
    assert R.shape == (3, 3)
    transform = np.concatenate([R, C[:, None]], axis=-1)
    return torch.tensor(transform, dtype=pose1.dtype)


def interp3(pose1, pose2, pose3, s12, s3):
    return interp(interp(pose1, pose2, s12).cpu(), pose3, s3)
