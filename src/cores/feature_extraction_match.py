import cv2 as cv
import numpy as np


def extract_SIFT_features(img):
    sift = cv.SIFT_create()
    # sift = cv.SIFT_create(
    #     nfeatures=2000, 
    #     nOctaveLayers=5,         
    #     contrastThreshold=0.04,  
    #     edgeThreshold=10        
    # )
    kps, descs = sift.detectAndCompute(img, None)
    return kps, descs


def match_SIFT_features(descs1, descs2, kps1, kps2):
    pts1 = []
    pts2 = []
    good_matches = []

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descs1, descs2, k=2)

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            pts1.append(kps1[m.queryIdx].pt)
            pts2.append(kps2[m.trainIdx].pt)
            good_matches.append(m)

    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)
    return pts1, pts2, good_matches
