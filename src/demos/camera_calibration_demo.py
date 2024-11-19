import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
h, w = 6, 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0)...,(h-1,w-1,0)
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

from config import DATA_CALIBRATION_DIR_PATH, OUT_DEMO_DIR_PATH

if not os.path.exists(OUT_DEMO_DIR_PATH):
    os.makedirs(OUT_DEMO_DIR_PATH)

calibration_img_paths = glob.glob(os.path.join(DATA_CALIBRATION_DIR_PATH, 'chessboard??.jpg'))
print(f"num of calibration images: {len(calibration_img_paths)}")


def calibrate_camera():
    for img_path in calibration_img_paths:
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (h, w), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (h, w), corners2, ret)
            result_img_name = "calibration_" + os.path.basename(img_path)
            result_img_path = os.path.join(OUT_DEMO_DIR_PATH, result_img_name)
            cv.imwrite(result_img_path, img)
        else:
            print(f"Chessboard corners not found in {img_path}")

    ret, intrinsic_mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, intrinsic_mtx, dist, rvecs, tvecs


ret, intrinsic_mtx, dist, rvecs, tvecs = calibrate_camera()
print("Calibration Result: ", ret)
print("Intrinsic Matrix: ", intrinsic_mtx)
print("Distortion Coefficients: ", dist)
print("Rvecs: ", rvecs)
print("Tvecs: ", tvecs)

np.savez(os.path.join(OUT_DEMO_DIR_PATH, 'demo_calibration_result.npz'), intrinsic_mtx=intrinsic_mtx, dist=dist,
         rvecs=rvecs, tvecs=tvecs)
