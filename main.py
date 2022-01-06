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
from torchvision import transforms
def decode(patch):
    transform = transforms.Compose([transforms.Normalize(mean=(0.,0.,0.), std=(1/0.229, 1/0.224, 1/0.225)),
                                transforms.Normalize(mean=(-0.485, -0.456, -0.406), std=(1.,1.,1.))])
    patch = transform(patch)
    patch = patch.permute(1, 2, 0).cpu().numpy()
    return patch

if __name__ == '__main__':

    human_detector = DetectNet(use_cuda=True)
    pose_estimator = PoseEstimatorV2(use_cuda=True)
    data_folder = '/Data/3D_pose_estimation_dataset/PIROPO/Room A/omni_1A/omni1A_test6'
    data_folder = '/Data/3D_pose_estimation_dataset/CEPDOF/Edge_cases'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS20'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuCo/data/unaugmented_set/1'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS1'
    # data_folder = '/Data/3D_pose_estimation_dataset/RAPID'
    vis = True
    if __name__ == '__main__':
        for fn in sorted(os.listdir(data_folder)):
            if 'jpg' not in fn and 'png' not in fn:
                continue
            path = os.path.join(data_folder, fn)
            frame = cv2.imread(path)
            start = time.time()
            patches, k_values = human_detector.detect(frame)

            patches = torch.stack(patches, 0).cuda()
            patches *= 255
            patches = pose_estimator.transform(patches)
            k_values = torch.stack(k_values, 0).unsqueeze(-1).cuda()
            pose = pose_estimator.batch_forward(patches, k_values)
            if vis:
                patch = patches[0]
                patch = decode(patch)
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                patch = np.ascontiguousarray(patch, dtype=np.uint8)
                pose = pose[0].cpu().numpy()
                tmpimg = pose_estimator.visualize(patch, pose)
                cv2.imshow('', tmpimg)
                cv2.waitKey()

            # for patch, k_value in zip(patches, k_values):
                # patch is RGB image in the range of (0,1)
                # k_value = k_value.unsqueeze(0)
                # patch *= 255
                # pose = pose_estimator.forward(patch, k_value)


            print(time.time()-start)




