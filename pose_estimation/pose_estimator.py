# TODO: Do 3D Human Pose Estimation from given images.
import cv2
from connection.server import Server
import torch
import config_test
from config import cfg
from pose_estimation.MobileHumanPose.main.model import get_pose_net
from torch.nn.parallel.data_parallel import DataParallel
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from utils.vis import vis_keypoints

import os
class PoseEstimator:
    def __init__(self):
        # TODO: initialize models
        self.model = get_pose_net('LPSKI', False, 21)
        self.model = DataParallel(self.model).cuda()
        checkpoint_path = '/Data/MobileHumanPose/output/model_dump/snapshot_7.pth.tar'
        ckpt = torch.load(checkpoint_path)
        self.model.load_state_dict(ckpt['network'])
        self.model.eval()

    def forward(self, image):
        # TODO: process image to 3d skeleton
        image = cv2.resize(image, (cfg.input_shape[1], cfg.input_shape[0]))
        image = Image.fromarray(image)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
        image = transform(image).cuda()[None, :, :, :]
        print(image.shape)
        with torch.no_grad():
            pose = self.model(image)
        return pose

    def visualize(self, image, pose):
        joint_num = 21
        skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

        # tmpimg = image[0].cpu().numpy()
        # tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3, 1, 1) + np.array(cfg.pixel_mean).reshape(3, 1, 1)
        # tmpimg = tmpimg.astype(np.uint8)
        # tmpimg = tmpimg[::-1, :, :]
        # tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()
        tmpkps = np.zeros((3, joint_num))
        tmpkps[:2, :] = pose[0, :, :2].cpu().numpy().transpose(1, 0) / cfg.output_shape[0] * cfg.input_shape[0]
        tmpkps[2, :] = 1
        tmpimg = vis_keypoints(image, tmpkps, skeleton)
        return tmpimg


if __name__ == '__main__':
    pose_estimator = PoseEstimator()
    piropo_folder = '/Data/3D_pose_estimation_dataset/PIROPO/Room A/omni_1A/omni1A_test6'
    # piropo_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS1'

    if __name__ == '__main__':
        for fn in sorted(os.listdir(piropo_folder)):
            if 'jpg' not in fn:
                continue
            path = os.path.join(piropo_folder, fn)
            frame = cv2.imread(path)
            # TODO: feed bounding box of a person instead of a full image.
            pose = pose_estimator.forward(frame)
            print(pose)
            pose_frame = pose_estimator.visualize(frame, pose)
            cv2.imshow('', pose_frame)
            cv2.waitKey(1)


    # If the server recieves frames from other process use below code.
    # server = Server()
    # pose_estimation = PoseEstimator()
    # while True:
    #     frame = server.get_image()
    #     pose = pose_estimation.forward(frame)
    #     print(pose.shape)
