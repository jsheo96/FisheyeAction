# main.py
# execute our program with various options given by arguments
from pose_estimation.utils.vis import vis_3d_multiple_skeleton
import numpy as np
from pose_estimation.config import cfg
from pose_estimation.pose_estimator import PoseEstimatorV2
import os
import cv2
import sys
sys.path.insert(0, 'human_detection')

from human_detection.detector import DetectNet
import torch
import time
if __name__ == '__main__':

    human_detector = DetectNet(use_cuda=True)
    pose_estimator = PoseEstimatorV2(use_cuda=True)
    data_folder = '/Data/3D_pose_estimation_dataset/PIROPO/Room A/omni_1A/omni1A_test6'
    data_folder = '/Data/3D_pose_estimation_dataset/CEPDOF/Edge_cases'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS20'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuCo/data/unaugmented_set/1'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS1'
    # data_folder = '/Data/3D_pose_estimation_dataset/RAPID'
    if __name__ == '__main__':
        for fn in sorted(os.listdir(data_folder)):
            if 'jpg' not in fn and 'png' not in fn:
                continue
            path = os.path.join(data_folder, fn)
            frame = cv2.imread(path)
            patches, k_values = human_detector.detect(frame)
            start = time.time()
            for patch,k_value in zip(patches, k_values):
                patch = patch.permute(1,2,0).cpu().numpy()
                k_value = k_value.unsqueeze(0)
                pose = pose_estimator.forward(patch, k_value)
                #patch = (patch * 255).astype(np.uint8)
                ##tmpimg = pose_estimator.visualize(patch, pose)
                #print(tmpimg.dtype,tmpimg.shape)
                #cv2.imshow('',tmpimg)
                #cv2.waitKey()
            print('elapsed time to process: {}'.format(time.time()-start))




