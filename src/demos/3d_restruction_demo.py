import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import DATA_DIR_PATH, OUT_DIR_PATH, DATA_CALIBRATION_DIR_PATH

import numpy as np
import cv2 as cv

OUT_DEMO_DIR_PATH = os.path.join(OUT_DIR_PATH, 'demo')
if not os.path.exists(OUT_DEMO_DIR_PATH):
    os.makedirs(OUT_DEMO_DIR_PATH)

"""
pose estimation: PnP: 3D object points to 2D image points
"""
# load previously saved calibration result
with np.load(os.path.join(OUT_DEMO_DIR_PATH, 'demo_calibration_result.npz')) as X:
    intrinsic_mtx, dist, _, _ = [X[i] for i in ('intrinsic_mtx', 'dist', 'rvecs', 'tvecs')]

print("intrinsic_mtx: \n", intrinsic_mtx)
print("dist: \n", dist)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, np.int32(corner), np.int32(tuple(imgpts[0].ravel())), (255, 0, 0), 5)
    img = cv.line(img, np.int32(corner), np.int32(tuple(imgpts[1].ravel())), (0, 255, 0), 5)
    img = cv.line(img, np.int32(corner), np.int32(tuple(imgpts[2].ravel())), (0, 0, 255), 5)
    return img


h, w = 6, 9

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
print("axis: \n", axis)

import glob

chessboard_img_paths = glob.glob(os.path.join(DATA_CALIBRATION_DIR_PATH, 'chessboard0[1-3].jpg'))

for img_path in chessboard_img_paths:
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (h, w), None)

    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, intrinsic_mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, intrinsic_mtx, dist)

        img = draw(img, corners2, imgpts)
        cv.imwrite(os.path.join(OUT_DEMO_DIR_PATH, 'pose_estimation_' + os.path.basename(img_path)), img)
        # cv.namedWindow('img', cv.WINDOW_NORMAL)
        # cv.resizeWindow('img', 800, 600)
        # cv.imshow('img', img)
        # k = cv.waitKey(0) & 0xFF

cv.destroyAllWindows()

"""
Epipolar: Fundamental matrix, Essential matrix, R, T.
TraingulatePoints: 2D feature points to reconstruct 3D points
"""
from matplotlib import pyplot as plt

DATA_DEMO_DIR_PATH = os.path.join(DATA_DIR_PATH, 'demo')

demo_img_paths = glob.glob(os.path.join(DATA_DEMO_DIR_PATH, 'chair*.jpg'))

img1 = cv.imread(demo_img_paths[0], 0)  # queryimage # left image
img2 = cv.imread(demo_img_paths[1], 0)  # trainimage # right image

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC, 1.0, 0.99)
E, mask = cv.findEssentialMat(pts1, pts2, intrinsic_mtx, cv.RANSAC, 0.99, 1.0)

E_indirect = intrinsic_mtx.T @ F @ intrinsic_mtx
U, S, Vt = np.linalg.svd(E_indirect)
S = S / S[0]  # 将最大奇异值归一化为1
S = np.diag([1, 1, 0])  # 强制本质矩阵的代数性质
E_indirect = U @ S @ Vt

E = E / np.linalg.norm(E, 'fro')
E_indirect = E_indirect / np.linalg.norm(E_indirect, 'fro')

print("F: \n", F)
print("E: \n", E)
print("E from F:\n", E_indirect)


def check_essential_properties(E, name=""):
    """检查本质矩阵的性质"""
    # 检查奇异值
    U, S, Vt = np.linalg.svd(E)

    # 计算Frobenius范数
    frob_norm = np.linalg.norm(E, 'fro')

    print(f"\n{name} Essential Matrix properties:")
    print(f"Singular values: {S}")
    print(f"Frobenius norm: {frob_norm}")
    print(f"Ratio of first two singular values: {S[0] / S[1]}")
    print(f"Third singular value: {S[2]}")


check_essential_properties(E, "Direct")

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

cv.imwrite(os.path.join(OUT_DEMO_DIR_PATH, 'epipolar_' + os.path.basename(demo_img_paths[0])), img5)
cv.imwrite(os.path.join(OUT_DEMO_DIR_PATH, 'epipolar_' + os.path.basename(demo_img_paths[1])), img3)

# plt.figure(figsize=(12, 6))
# plt.subplot(121), plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
# plt.subplot(122), plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
# plt.show()
