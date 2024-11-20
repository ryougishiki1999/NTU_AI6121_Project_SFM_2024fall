import numpy as np
import os
import random

import cv2 as cv

from config import LOG_VERBOSE, OUT_DIR_PATH
from cores.camera_calibration import CameraCalibrator
from cores.feature_extraction_match import extract_SIFT_features, match_SIFT_features
from cores.init_reconstruction import InitReconstructor
from cores.bundle_adjustment import GlobalBundleAdjustmentSolver
from utils.load_sfm_img import SFMImgLoader
from utils.visualize_sfm_results import SFMResultVisualizer
from utils.points3D_update import update_good_matches_dict, get_all_unique_points3D, update_unique_points3D_dict, match_3d_by_2d
from utils.camera_poses import generate_camera_pose, recover_camera_pose, generate_projection_matrix

class SFMEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SFMEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if not os.path.exists(OUT_DIR_PATH):
            os.makedirs(OUT_DIR_PATH)

        self.camera_poses = dict()
        # shape is (N, 3), N is the number of 3D points
        self.good_matches_dict = dict()
        self.unqiue_points3D_dict = dict()
        self.last_img_idx = -1

        self.camera_calibrator = CameraCalibrator()
        self.K, self.dist = self.camera_calibrator.load_calibration_result()

        self.sfm_img_loader = SFMImgLoader()
        #self.imgs, self.K = self.sfm_img_loader.get_sfm_undistorted_imgs(self.K, self.dist)
        self.imgs = self.sfm_img_loader.get_sfm_imgs()
        self.global_ba_solver = GlobalBundleAdjustmentSolver(self.K)
        self.init_reconstructor = InitReconstructor(self.K)
        
        self.sfm_result_visualizer = SFMResultVisualizer()
        
        

    def _select_next_img(self):
        self.last_img_idx = max(self.camera_poses.keys())
        next_idx = self.last_img_idx + 1
        if next_idx >= len(self.imgs):
            return self.last_img_idx, None
        else:
            return self.last_img_idx, next_idx
        
    def _incremental_reconstruction(self, query_idx, train_idx):
        query_img = self.imgs[query_idx]
        train_img = self.imgs[train_idx]
        
        query_kpts, query_pts_desc = extract_SIFT_features(query_img)
        train_kpts, train_pts_desc = extract_SIFT_features(train_img)
        
        query_pts, train_pts, good_matches = match_SIFT_features(query_pts_desc, train_pts_desc, query_kpts, train_kpts)
        update_good_matches_dict(self.good_matches_dict, train_idx, train_pts, query_pts)
        print("Number of good matches", len(good_matches))
        
        total_points3D, matched_mask, incremental_mask =\
            match_3d_by_2d(self.unqiue_points3D_dict, self.good_matches_dict, query_idx, query_pts)
            
        matched_points3D = total_points3D[matched_mask]
        matched_train_pts = train_pts[matched_mask]
        
        print(f"Number of correspondence between existed 3Dpoints and 2D feature points from img{train_idx}: ", np.count_nonzero(matched_mask))
        print(f"Number of incremental 2D feature points from img{train_idx}: ", np.count_nonzero(incremental_mask))
        

        incremental_query_pts, incremental_train_pts = query_pts[incremental_mask], train_pts[incremental_mask]
        
        _, train_rvec, train_tvec, _ = cv.solvePnPRansac(matched_points3D, matched_train_pts, self.K, None, confidence=0.999)
        query_rvec, query_tvec = recover_camera_pose(self.camera_poses[query_idx])
        
        if LOG_VERBOSE:
            matched_query_pts = query_pts[matched_mask]
            _, val_query_rvec, val_query_tvec, _ = cv.solvePnPRansac(matched_points3D, matched_query_pts, self.K, None, confidence=0.999)
            print("val query rvec, tvec: ", val_query_rvec, val_query_tvec)
            print("query rvec, tvec: ", query_rvec, query_tvec)
        
        self.camera_poses[train_idx] = generate_camera_pose(train_rvec, train_tvec)
        print(f"camera pose of img {train_idx}: ", cv.Rodrigues(train_rvec)[0], train_tvec)
        
        
        P_query = generate_projection_matrix(query_rvec, query_tvec, self.K)
        P_train = generate_projection_matrix(train_rvec, train_tvec, self.K)
        
        new_Points4D = cv.triangulatePoints(P_query, P_train, incremental_query_pts.T, incremental_train_pts.T)
        new_Points3D = new_Points4D[:3] / new_Points4D[3]
        new_Points3D = new_Points3D.T
        
        unique_mask = update_unique_points3D_dict(self.unqiue_points3D_dict, train_idx, new_Points3D, incremental_train_pts)

        print("number of incremental points3D: ", new_Points3D.shape[0])
        print("number of unique incremental points3D: ", new_Points3D[unique_mask].shape[0])
    
    def _incremental_reconstructions(self):
        while True:
            last_idx, nxt_idx = self._select_next_img()
            print("\nQuery img idx: ", last_idx, "Train img idx: ", nxt_idx)
            if nxt_idx is None:
                break # no more images to add into incremental reconstructions
            
            self._incremental_reconstruction(last_idx, nxt_idx)
    
    def _global_bundle_adjustment(self):
        self.global_ba_solver.set_data(self.camera_poses, self.unqiue_points3D_dict)    
        optimized_poses, optimized_points3D_dict = self.global_ba_solver.optimize()
        self.camera_poses = optimized_poses
        self.unqiue_points3D_dict = optimized_points3D_dict
        
    def filter_points_iqr(self, points3D, k=3.0):
        centroid = np.mean(points3D, axis=0)
        distances = np.linalg.norm(points3D - centroid, axis=1)
        
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1

        threshold = Q3 + k * IQR
        
        mask = distances < threshold
        filtered_points = points3D[mask]
        
        print(f"original number of points3D: {len(points3D)}")
        print(f"filterd number of points3D: {len(filtered_points)}")
        print(f"distance threshold: {threshold:.2f}")
        print(f"removed number of points3D: {len(points3D) - len(filtered_points)}")
        
        return filtered_points
    

    def run(self):
        print("SFM Engine is running...")
        print("Number of sfm images: ", len(self.imgs))
        print("Camera Intrinsic Matrix: \n", self.K)
        print("Camera Distortion Coefficients: \n", self.dist)
        camera_poses, good_matches_dict, unqiue_points3D_dict = \
            self.init_reconstructor.run(self.imgs)
        self.camera_poses.update(camera_poses)
        self.good_matches_dict.update(good_matches_dict)
        self.unqiue_points3D_dict.update(unqiue_points3D_dict)
    
        print("After initial reconstruction, number of 3D points: ", get_all_unique_points3D(self.unqiue_points3D_dict).shape[0])
        print("After initial reconstruction, number of camera poses: ", len(self.camera_poses))
        if LOG_VERBOSE:
            print("After initial reconstruction, camera poses: ", self.camera_poses)
            print("After initial reconstruction, good matches dict example: ", random.sample(list(self.good_matches_dict[1]), k=5))
            print("After initial reconstruction, unique points3D dict example: ", random.sample(list(self.unqiue_points3D_dict[1]), k=5))
        print("\n=============================================================")
        print("Start incremental reconstructions...")
        self._incremental_reconstructions()
        print("End incremental reconstructions...")
        print("=============================================================\n")
        all_unique_points3D = get_all_unique_points3D(self.unqiue_points3D_dict)
        print("After incremental reconstructions, number of 3D points: ", all_unique_points3D.shape[0])
        if LOG_VERBOSE:
            print("After incremental reconstructions, number of camera poses: ", len(self.camera_poses))
            print("After incremental reconstructions, length of good matches dict: ", len(self.good_matches_dict))
            print("After incremental reconstructions, length of unique incremental points3D dict: ", len(self.unqiue_points3D_dict))
        #self._global_bundle_adjustment()
        filtered_points3D = self.filter_points_iqr(all_unique_points3D)
        print("After global bundle adjustment, number of 3D points: ", filtered_points3D.shape[0])
        self.sfm_result_visualizer.run(filtered_points3D, self.camera_poses)
        print("SFM Engine is done.")
