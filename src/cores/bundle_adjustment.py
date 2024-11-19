import numpy as np
import cv2 as cv
from scipy.optimize import least_squares

class GlobalBundleAdjustmentSolver:
    _instance = None
    
    def __new__(cls, K):
        if cls._instance is None:
            cls._instance = super(GlobalBundleAdjustmentSolver, cls).__new__(cls)
            cls._instance._initialize(K)
        return cls._instance
    
    def _initialize(self, K):
        self.K = K
        self.camera_poses = None # {img_idx: [rx, ry, rz, tx, ty, tz]}
        self.unique_points3d_dict = None # {img_idx: {point2D: point3D}}
    
    def _project_points(self, points3d, pose):
        points2d, _ = cv.projectPoints(points3d, pose[:3], pose[3:], self.K, None)
        return points2d.reshape(-1, 2)
    
    def _compute_residuals(self,x):
        # Get number of camera poses
        n_poses = len(self.camera_poses)
        poses = np.zeros((n_poses, 6))
        poses[0] = list(self.camera_poses.values())[0]  # fix first pose
        poses[1:] = x[:(n_poses-1) * 6].reshape(-1, 6)
        points3d = x[(n_poses-1) * 6:].reshape(-1, 3)
        
        residuals = []
        point_idx = 0
        
        for img_idx, points2d_dict in self.unique_points3d_dict.items():
            # Get number of points in current image
            n_points = len(points2d_dict)
            # Project 3D points to current image
            proj = self._project_points(points3d[point_idx:point_idx + n_points], poses[img_idx])
            # Get observed 2D points
            obs = np.array(list(points2d_dict.keys()))
            # Calculate residuals (projected - observed)
            residuals.append((proj - obs).ravel())
            # Update point index
            point_idx += n_points
        
        return np.concatenate(residuals)

    def set_data(self, camera_poses, unique_points3d_dict):
        self.camera_poses = camera_poses
        self.unique_points3d_dict = unique_points3d_dict
    
    def optimize(self):
        # Convert camera poses to array
        poses_array = np.array(list(self.camera_poses.values()))
        # Convert all 3D points to array
        points3d_array = np.array([p for points in self.unique_points3d_dict.values() for p in points.values()])
        
        # Construct initial optimization variables
        x0 = np.concatenate([poses_array[1:].ravel(), points3d_array.ravel()])
        
        # Optimizing using Levenberg-Marquardt algorithm
        #result = least_squares(self._compute_residuals, x0, method='trf',loss='soft_l1', verbose=2)
        result = least_squares(self._compute_residuals, x0, verbose=2)
        
        # Extract optimized camera poses and 3D points
        n_poses = len(self.camera_poses)
        opt_poses = np.zeros((n_poses, 6))
        opt_poses[0] = list(self.camera_poses.values())[0]
        opt_poses[1:] = result.x[:(n_poses-1) * 6].reshape(-1, 6)
        # Extract optimized 3D points
        opt_points = result.x[n_poses * 6:].reshape(-1, 3)
        
        point_idx = 0    
        for i, img_idx in enumerate(self.unique_points3d_dict.keys()):
            self.camera_poses[img_idx] = opt_points[i]
            # Update unique points3D dict for current image
            n_points = len(self.unique_points3d_dict[img_idx])
            for (p2d, _), p3d in zip(self.unique_points3d_dict[img_idx].items(), opt_points[point_idx:point_idx + n_points]):
                self.unique_points3d_dict[img_idx][p2d] = p3d
            point_idx += n_points
            
        return self.camera_poses, self.unique_points3d_dict
        
