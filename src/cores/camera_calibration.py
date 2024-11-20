import glob
import os
from tkinter import E

import cv2 as cv
import numpy as np

from config import DATA_CALIBRATION_DIR_PATH, OUT_CALIBRATION_DIR_PATH, DATA_CALIBRATION_RESULT_FILE_PATH, LOG_VERBOSE

class CameraCalibrator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CameraCalibrator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):

        if not os.path.exists(OUT_CALIBRATION_DIR_PATH):
            os.makedirs(OUT_CALIBRATION_DIR_PATH)

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.pattern_size = (9, 6)
        h, w = self.pattern_size
        self.objps = np.zeros((w * h, 3), np.float32)
        self.objps[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

        self.pattern_img_paths = glob.glob(os.path.join(DATA_CALIBRATION_DIR_PATH, 'chessboard*.jpg'))
        print(f"{__file__}: number of loaded calibration pattern images: {len(self.pattern_img_paths)}")
        self.n_pattern_imgs = len(self.pattern_img_paths)
        if self.n_pattern_imgs == 0:
            raise Exception("No pattern images found in the calibration directory.")

        # length of objps_list and imgps_list will be equal to n_pattern_images
        # after running run_calibration method
        self.objps_list = []
        self.imgps_list = []

        self.K, self.dist = None, None

    def _run_calibration_and_save_result(self):
        print("Running calibration and saving result...")
        for pattern_img_path in self.pattern_img_paths:
            pattern_img = cv.imread(pattern_img_path)
            gray_pattern_img = cv.cvtColor(pattern_img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray_pattern_img, self.pattern_size, None)

            if ret == True:
                self.objps_list.append(self.objps)

                corners2 = cv.cornerSubPix(gray_pattern_img, corners, (11, 11), (-1, -1), self.criteria)
                self.imgps_list.append(corners2)

                cv.drawChessboardCorners(pattern_img, self.pattern_size, corners2, ret)
                calibration_pattern_img = "calibration_" + os.path.basename(pattern_img_path)
                calibration_pattern_img_path = os.path.join(DATA_CALIBRATION_DIR_PATH, calibration_pattern_img)
                cv.imwrite(calibration_pattern_img_path, pattern_img)
            else:
                print(f"Chessboard corners not found in {pattern_img_path}")

        ret, self.K, self.dist, _, _ = cv.calibrateCamera(self.objps_list, self.imgps_list,
                                                          gray_pattern_img.shape[::-1], None, None)
        if ret:
            np.savez(DATA_CALIBRATION_RESULT_FILE_PATH, intrinsic_mtx=self.K, dist=self.dist)
        else:
            print("Calibration failed. Some errors occurred.")

    def _is_result_exist(self):
        return True if os.path.isfile(DATA_CALIBRATION_RESULT_FILE_PATH) else False

    def load_calibration_result(self):
        if not self._is_result_exist():
            self._run_calibration_and_save_result()

        with np.load(DATA_CALIBRATION_RESULT_FILE_PATH) as X:
            self.K, self.dist = [X[i] for i in ('intrinsic_mtx', 'dist')]
            
        print(f"{__file__}: load Camera Intrinsic Matrix: \n", self.K)
        print(f"{__file__}: load Camera Distortion Coefficients: \n", self.dist)
        return self.K, self.dist
