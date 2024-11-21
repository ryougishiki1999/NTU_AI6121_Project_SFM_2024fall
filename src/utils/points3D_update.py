import numpy as np
from scipy.spatial import cKDTree

from config import LOG_VERBOSE


def update_good_matches_dict(good_matches_dict, train_img_idx, train_pts, query_pts):
    """_summary_

    Args:
        good_matches_dict (_type_): key:trainImg idx, value: dict{key: trainImg pt, value: queryImg pt}, 
        trainImg idx from 1 to N_SFM_IMGS-1
    """
    if train_img_idx not in good_matches_dict:
        good_matches_dict[train_img_idx] = dict()
        
    for train_pt, query_pt in zip(train_pts, query_pts):
        good_matches_dict[train_img_idx][tuple(train_pt)] = tuple(query_pt)
        
def get_all_unique_points3D(unique_points3D_dict):
    all_unique_points3D = None
    for _, unqiue_point3D_dict in unique_points3D_dict.items():
        for _, unique_point3D in unqiue_point3D_dict.items():
            if all_unique_points3D is None:
                all_unique_points3D = unique_point3D
            else:
                all_unique_points3D = np.vstack((all_unique_points3D, unique_point3D))
    return all_unique_points3D


def init_unqiue_points3D_dict(unique_points3D_dict, points3D, pts, train_idx, tolerance=1e-10):
    mask = np.ones(points3D.shape[0], dtype=bool)

    for i in range(points3D.shape[0]):
        if mask[i]:
            for j in range(i + 1, points3D.shape[0]):
                if mask[j]:
                    if np.all(np.abs(points3D[i] - points3D[j]) < tolerance):
                        mask[j] = False

    unique_points3D = points3D[mask]
    unique_pts = pts[mask]
    
    if train_idx not in unique_points3D_dict:
        unique_points3D_dict[train_idx] = dict()
    
    for unique_pt, unique_point3D in zip(unique_pts, unique_points3D):
        unique_points3D_dict[train_idx][tuple(unique_pt)] = np.array(unique_point3D,dtype=np.float64)
        
def update_unique_points3D_dict(unqiue_points3D_dict, train_img_idx, new_points3D, new_pts, distance_threshold=1e-2):
    print("update unique points3D dict, train_img_idx: ", train_img_idx)
    all_unique_points3D = get_all_unique_points3D(unqiue_points3D_dict)

    unique_mask = np.ones(new_points3D.shape[0], dtype=bool)
    
    all_unique_points3D = np.ascontiguousarray(all_unique_points3D)
    new_points3D = np.ascontiguousarray(new_points3D)
    tree = cKDTree(all_unique_points3D)
    distances, _ = tree.query(new_points3D, k=1)
    unique_mask = distances > distance_threshold

    
    # for i, new_point3D in enumerate(new_points3D):
    #     for _, unique_point3D in enumerate(all_unique_points3D):
    #         if np.array_equal(new_point3D, unique_point3D):
    #             unique_mask[i] = False
                
    new_unqiue_points3D = new_points3D[unique_mask]
    new_unqiue_pts = new_pts[unique_mask]
    
    if train_img_idx not in unqiue_points3D_dict:
        unqiue_points3D_dict[train_img_idx] = dict()
    
    for unique_pt, unique_point3D in zip(new_unqiue_pts, new_unqiue_points3D):
        unqiue_points3D_dict[train_img_idx][tuple(unique_pt)] = np.array(unique_point3D, dtype=np.float64)
        
    return unique_mask

def match_3d_by_2d(unique_points3D_dict, good_matches_dict, src_idx, src_pts):
    """
    Find all corresponding 3D points for the 2D points in src_pts.

    Args:
        unique_points3D_dict : {key: image index, value: dict{key: 2D point, value: 3D point}}
        good_matches_dict (_type_): {key: image index, value: dict{key: trainImg pt, value: queryImg pt}}
        src_idx (_type_): source image index
        src_pts (_type_): source 2D points
    """
                
    def index_start_pt_on_target_idx(src_pt, target_idx):
        """
            find whether the src_pt can be indexed on target_idx, if so, return the indexed point.
            if not, return None.
        Args:
            src_pt : source 2D point
            target_idx : target image index
        """
        
        def recursive_indexing(pt, idx):
            """
            find whether the point is in the good_matches_dict[idx], if so, return the indexed point.
            if not, return None.

            Args:
                pt: 2D point
                idx: image index
            """
            if idx == target_idx:
                indexed_pt = pt
                return indexed_pt
            else:
                good_matches = good_matches_dict[idx]
                if tuple(pt) in good_matches:
                    indexed_pt = good_matches[tuple(pt)]
                    nxt_idx = idx - 1
                    return recursive_indexing(indexed_pt, nxt_idx)
                else:
                    return None
                
        return recursive_indexing(src_pt, src_idx)
    
    def match_on_target_idx(target_idx):
        """
        find whether the src_pts have corresponding 3D points in the unique_points3D_dict[target_idx]

        Args:
            target_idx : target image index
        """
        unique_point3D_dict = unique_points3D_dict[target_idx]
        for i, start_pt in enumerate(src_pts):
            if matched_mask[i]:
                continue
            else:
                indexed_pt = index_start_pt_on_target_idx(start_pt, target_idx)
                if indexed_pt is not None and tuple(indexed_pt) in unique_point3D_dict:
                    matched_mask[i] = True
                    incremental_mask[i] = False
                    total_points3D[i] = unique_point3D_dict[tuple(indexed_pt)]
                
    matched_mask = np.zeros(src_pts.shape[0], dtype=bool)
    incremental_mask = np.ones(src_pts.shape[0], dtype=bool)
    total_points3D = np.zeros((src_pts.shape[0], 3), dtype=np.float64)
    
    # Correspondent Search from image src_idx to 1, 1 is the first image.
    cur_idx, end_idx = src_idx, 1
    while cur_idx >= end_idx:
        match_on_target_idx(cur_idx)
        cur_idx -= 1
    
    return total_points3D, matched_mask, incremental_mask
    