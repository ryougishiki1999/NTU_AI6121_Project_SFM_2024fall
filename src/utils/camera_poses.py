import numpy as np
import cv2 as cv

def generate_camera_pose(rvec, tvec):
    """_summary_

    Args:
        rvec (_type_): (3,1)
        tvec (_type_): (3,1)
    """
    camera_pose = np.concatenate([rvec.flatten(), tvec.flatten()]) # (6,) [rx, ry, rz, tx, ty, tz]
    print("tmp camera pose: ", camera_pose)
    return camera_pose

def recover_camera_pose(camera_pose):
    rvec = camera_pose[:3].reshape(3,1)
    tvec = camera_pose[3:].reshape(3,1)
    return rvec, tvec

def generate_projection_matrix(rvec, tvec, K):
    """_summary_

    Args:
        camera_matrix (_type_): (3,3)
        rvec (_type_): (3,1)
        tvec (_type_): (3,1)
    """
    R = cv.Rodrigues(rvec)[0]
    Rt = np.concatenate([R, tvec], axis=1)
    P = K @ Rt
    return P



