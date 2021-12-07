# main.py
# execute our program with various options given by arguments
from pose_estimation.utils.vis import vis_3d_multiple_skeleton
import numpy as np
from pose_estimation.config import cfg
from pose_estimation.pose_estimator import PoseEstimator
import os
import cv2
from human_detection.detector import DetectNet
if __name__ == '__main__':
    human_detector = DetectNet()
    pose_estimator = PoseEstimator()
    data_folder = '/Data/3D_pose_estimation_dataset/PIROPO/Room A/omni_1A/omni1A_test6'
    data_folder = '/Data/3D_pose_estimation_dataset/CEPDOF/Edge_cases'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS20'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuCo/data/unaugmented_set/1'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS1'
    # data_folder = '/Data/3D_pose_estimation_dataset/RAPID'
    # save_folder = '/Data/3D_pose_estimation_dataset/demo_result'
    if __name__ == '__main__':
        for fn in sorted(os.listdir(data_folder)):
            if 'jpg' not in fn and 'png' not in fn:
                continue
            path = os.path.join(data_folder, fn)
            frame = cv2.imread(path)

            patches, k_values = human_detector.detect(frame)
            for patch in patches:
                cv2.imshow('patches[0]', patch.permute(1,2,0).cpu().numpy()[:,:,::-1])
            # TODO feed those patches to pose estimator
            # TODO get rid of bbox/keypoints transformation from pose_estimator
            # frame = cv2.resize(frame, (256, 256))
            poses, vis_kps = pose_estimator.forward(frame)
            # pickle.dump((poses, vis_kps), open(os.path.join(save_folder, fn.split('.')[0]+'.pkl'), 'wb'))
            for pose in poses:
                # print(pose)
                frame = pose_estimator.visualize(frame, pose)
            # vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), cfg.skeleton,
            #                          'output_pose_3d (x,y,z: camera-centered. mm.)')
            cv2.imshow('', cv2.resize(frame, None, fx=0.25, fy=0.25))
            cv2.waitKey()