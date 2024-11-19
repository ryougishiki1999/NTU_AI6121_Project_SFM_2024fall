import cv2 as cv
import numpy as np

from config import N_SFM_INTI_RECONSTRUCTION_PAIRS, LOG_VERBOSE
from cores.feature_extraction_match import extract_SIFT_features, match_SIFT_features
from utils.points3D_update import update_good_matches_dict, init_unqiue_points3D_dict, update_unique_points3D_dict
from utils.camera_poses import generate_camera_pose, recover_camera_pose, generate_projection_matrix


class InitReconstructor:
    _instance = None

    def __new__(cls, K):
        if cls._instance is None:
            cls._instance = super(InitReconstructor, cls).__new__(cls)
            cls._instance._initialize(K)
        return cls._instance

    def _initialize(self, K):
        self.K = K
        
        self.camera_poses = dict()
        R, tvec = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        rvec = cv.Rodrigues(R)[0]
        self.camera_poses[0] = generate_camera_pose(rvec, tvec)
        
        self.good_mathces_dict = dict()
        self.unique_points3D_dict = dict()

    def _select_consecutive_paris(self, imgs, n_pairs=N_SFM_INTI_RECONSTRUCTION_PAIRS):
        consecutive_paris = []
        for i in range(min(n_pairs, len(imgs) - 1)):
            consecutive_paris.append((imgs[i], imgs[i + 1], i, i + 1))
        return consecutive_paris

    def _reconstruct_by_consecutive_pair(self, consecutive_pair):
        img1, img2, img1_idx, img2_idx = consecutive_pair

        kp1, desc1 = extract_SIFT_features(img1)
        kp2, desc2 = extract_SIFT_features(img2)

        pts1, pts2, _ = match_SIFT_features(desc1, desc2, kp1, kp2)
        E, _ = cv.findEssentialMat(pts1, pts2, self.K, cv.RANSAC, 0.99, 1.0)
        
        update_good_matches_dict(self.good_mathces_dict, img2_idx, pts2, pts1)

        if LOG_VERBOSE:
            print(f"[{self.__class__.__name__}] parts of img{img1_idx} pts: \n", pts1[:5])
            print(f"[{self.__class__.__name__}] parts of img{img2_idx} pts: \n", pts2[:5])

        _, R, tvec, _ = cv.recoverPose(E, pts1, pts2, self.K)
        rvec1, tvec1 = recover_camera_pose(self.camera_poses[img1_idx])
        R1 = cv.Rodrigues(rvec1)[0]
        R2, tvec2 = R @ R1, R @ tvec1 + tvec
        rvec2 = cv.Rodrigues(R2)[0]
        self.camera_poses[img2_idx] = generate_camera_pose(rvec2, tvec2)
        if img1_idx == 0:
            print(f"{img1_idx} camera pose: ", R1, tvec1)
        print(f"{img2_idx} camera pose: ", R2, tvec2)
        
        P1 = generate_projection_matrix(rvec1, tvec1, self.K)
        P2 = generate_projection_matrix(rvec2, tvec2, self.K)

        points4D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points3D = points4D[:3] / points4D[3]
        points3D = points3D.T

        if LOG_VERBOSE:
            print(f"[{self.__class__.__name__}] number of points3D: ", points3D.shape[0])
            print(f"[{self.__class__.__name__}] parts of points3D triangulated from img{img1_idx} and img{img2_idx}:\n ",
                  points3D[0:5])

        return points3D, pts2

    def _reconstruct_by_consecutive_pairs(self, consecutive_pairs):
        for i, consecutive_pair in enumerate(consecutive_pairs):
            _, _, _, img2_idx = consecutive_pair
            new_points3D, new_pts = self._reconstruct_by_consecutive_pair(consecutive_pair)
            if i == 0:
                init_unqiue_points3D_dict(
                    self.unique_points3D_dict, new_points3D, new_pts, img2_idx
                )
            else:
                update_unique_points3D_dict(
                    self.unique_points3D_dict, img2_idx, new_points3D, new_pts 
                )
                

    def run(self, imgs):
        consecutive_pairs = self._select_consecutive_paris(imgs)
        if LOG_VERBOSE:
            print(f"[{self.__class__.__name__}] Number of consecutive_pairs: ", len(consecutive_pairs))
        self._reconstruct_by_consecutive_pairs(consecutive_pairs)
        return self.camera_poses, self.good_mathces_dict, self.unique_points3D_dict
