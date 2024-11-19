import os

import cv2 as cv

SRC_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR_PATH = os.path.dirname(SRC_DIR_PATH)
DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'data')
DATA_CALIBRATION_DIR_PATH = os.path.join(DATA_DIR_PATH, 'calibration')

DATA_CALIBRATION_RESULT_FILE_PATH = os.path.join(DATA_CALIBRATION_DIR_PATH, 'calibration_result.npz')

DATA_SFM_DIRS_PATH = os.path.join(DATA_DIR_PATH, 'sfm')
DATA_SFM_IMAGES_DIR_PATH = os.path.join(DATA_SFM_DIRS_PATH, 'leeweiname_lego', 'imgs')


RESOURCE_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'resource')
OUT_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'out')
OUT_CALIBRATION_DIR_PATH = os.path.join(OUT_DIR_PATH, 'calibration')
OUT_DEMO_DIR_PATH = os.path.join(OUT_DIR_PATH, 'demo')

LOG_VERBOSE = True
SFM_LOAD_SIZE = (1920, 1080) # (w, h)
SFM_CV_IMREAD_FLAG = cv.IMREAD_GRAYSCALE
# SFM_CV_IMREAD_FLAG = cv.IMREAD_COLOR
N_SFM_INTI_RECONSTRUCTION_PAIRS = 1 # min is 1
