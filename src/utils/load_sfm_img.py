import glob
import re
import os

import cv2 as cv

from config import DATA_SFM_IMAGES_DIR_PATH, SFM_CV_IMREAD_FLAG, SFM_LOAD_SIZE


class SFMImgLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SFMImgLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.sfm_imgs = []
        self.sfm_undistorted_imgs = []

    def _load_imgs(self, flags = SFM_CV_IMREAD_FLAG, load_size = SFM_LOAD_SIZE):
        
        def natural_sort_key(s):
            return int(re.search(r'sfm(\d+)', s).group(1))
            
        sfm_img_paths = sorted(glob.glob(os.path.join(DATA_SFM_IMAGES_DIR_PATH, 'sfm*.jpg')), key=natural_sort_key)
        for sfm_img_path in sfm_img_paths:
            print(f'Loading {sfm_img_path}')
            sfm_img = cv.imread(sfm_img_path, flags)
            # h, w = sfm_img.shape[:2]
            # if w > h: # if landscape
            #     sfm_img = cv.rotate(sfm_img, cv.ROTATE_90_CLOCKWISE)
            # sfm_img = cv.resize(sfm_img, load_size)
            self.sfm_imgs.append(sfm_img)

    def _load_undistort_imgs(self, K, dist, reference_size = SFM_LOAD_SIZE):
        if len(self.sfm_imgs) == 0:
            self._load_imgs()
            
        ref_w, ref_h,= reference_size
        new_K, roi = cv.getOptimalNewCameraMatrix(K, dist, (ref_w, ref_h), 1, (ref_w, ref_h))
        
        for sfm_img in self.sfm_imgs:
            h, w = sfm_img.shape[:2]

            # undistort
            undistorted_sfm_img = cv.undistort(sfm_img, K, dist, None, new_K)
            
            # crop the image
            x, y, w, h = roi
            undistorted_sfm_img = undistorted_sfm_img[y:y + h, x:x + w]
            self.sfm_undistorted_imgs.append(undistorted_sfm_img)
        return new_K

    def get_sfm_imgs(self):
        if len(self.sfm_imgs) == 0:
            self._load_imgs()
        return self.sfm_imgs

    def get_sfm_undistorted_imgs(self, K, dist):
        if len(self.sfm_undistorted_imgs) == 0:
            new_k = self._load_undistort_imgs(K, dist)
        return self.sfm_undistorted_imgs, new_k
