import pickle
# TODO: Do 3D Human Pose Estimation from given images.
import cv2
from connection.server import Server
import torch
from pose_estimation.config import cfg
from pose_estimation.mobilehumanpose import get_mobile_human_pose_net
from pose_estimation.rootnet import get_root_net
import torchvision.transforms as transforms
import numpy as np
from pose_estimation.utils.vis import vis_keypoints
import os
from pose_estimation.mask_rcnn import get_mask_rcnn
from pose_estimation.bbox_trans import gen_trans_from_patch_cv
from pose_estimation.utils.pose_utils import process_bbox, pixel2cam
from pose_estimation.dataset import generate_patch_image
from pose_estimation.dataset import gen_trans_from_patch_cv
from pose_estimation.utils.vis import vis_3d_keypoints
from pose_estimation.utils.vis import vis_3d_multiple_skeleton

import math
import time

class PoseEstimatorV2:
    def __init__(self):
        self.pose_net = get_mobile_human_pose_net()
        self.root_net = get_root_net()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])

    def forward(self, image_patch, k_value):
        assert image_patch.shape == (256, 256, 3), 'image_patch shape is not equal to (256, 256, 3). Got {}'.format(image_patch.shape)
        with torch.no_grad():
            image_patch = self.transform(image_patch)
            image_patch = image_patch.cuda()[None, :, :, :]
            pose = self.pose_net(image_patch)
            k_value = torch.tensor(k_value)
            root = self.root_net(image_patch, k_value)

            pose_3d = pose[0].cpu().numpy()
            pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + root[0, 2].item()
        return pose_3d

class PoseEstimator:
    def __init__(self):
        # TODO: initialize models
        self.pose_net = get_mobile_human_pose_net()
        self.root_net = get_root_net()
        self.detect_net = get_mask_rcnn()

    def forward(self, image):

        labels, bboxes = self.detect_net.detect(image)
        start = time.time()

        poses = []
        output_pose_3d_list = []
        height, width = image.shape[0], image.shape[1]
        focal = [1500.9799492788811, 1495.9003438753227]  # x-axis, y-axis
        # princpt = [width / 2, height / 2]  # x-axis, y-axis
        princpt = [1030.7205375378683, 1045.5236081955522]  # x-axis, y-axis
        for idx in (labels==1).nonzero():

            idx = idx.item()
            bbox = bboxes[idx]
            x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
            bbox = (x,y,w,h)
            bbox = process_bbox(bbox, width, height)

            image_human, img2bb_trans = generate_patch_image(image, bbox, False, 1.0, 0.0, False)

            # bbox = (x,y,w,h)dfdfdfdfd
            # bbox = process_bbox(bbox, width, height)
            # x1 = int(bbox[0]) if bbox[0] >= 0 else 0
            # y1 = int(bbox[1]) if bbox[1] >= 0 else 0
            # x2 = int(bbox[0]+bbox[2])
            # y2 = int(bbox[1]+bbox[3])
            # image_human = image[y1:y2, x1:x2, :]
            # cv2.imshow('pose_net input '+str(idx), image_human[:,:,::-1].astype(np.uint8))

            # image_human = cv2.resize(image, (cfg.input_shape[1], cfg.input_shape[0]))
            # image_human = image_human[:, :, ::-1].copy()
            # image_human = image_human.astype(np.float32)
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
            image_human = transform(image_human)
            image_human = image_human.cuda()[None, :, :, :]
            with torch.no_grad():
                pose = self.pose_net(image_human)
                # TODO: calculate k
                # TODO: get focal lengths of camera...
                area = w * h
                k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(area))]).astype(np.float32)
                # k_value = np.ones((image_human.shape[0], 1), dtype=np.float32)*3000.0
                k_value = torch.tensor(k_value)
                root = self.root_net(image_human, k_value)

                # TODO: pose post-processing by transformation matrix inverse
                pose_3d = pose[0].cpu().numpy()
                pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
                pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
                pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
                img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
                pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:,
                                 :2]
                pose = pose_3d[:, :2].copy()
                pose_3d[:, 2] = (pose_3d[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + root[0,2].item()
                pose_3d = pixel2cam(pose_3d, focal, princpt)
                output_pose_3d_list.append(pose_3d.copy())
                poses.append(pose)


        vis_kps = np.array(output_pose_3d_list)
        print(time.time() - start)

        return poses, vis_kps

    def visualize(self, image, pose):

        # tmpimg = image[0].cpu().numpy()
        # tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3, 1, 1) + np.array(cfg.pixel_mean).reshape(3, 1, 1)
        # tmpimg = tmpimg.astype(np.uint8)
        # tmpimg = tmpimg[::-1, :, :]
        # tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()

        # tmpkps = np.zeros((3, joint_num))
        # tmpkps[:2, :] = pose[0, :, :2].cpu().numpy().transpose(1, 0) / cfg.output_shape[0] * cfg.input_shape[0]
        # tmpkps[2, :] = 1

        vis_kps = np.zeros((3, cfg.joint_num))
        vis_kps[0, :] = pose[:, 0]
        vis_kps[1, :] = pose[:, 1]
        vis_kps[2, :] = 1

        tmpimg = vis_keypoints(image, vis_kps, cfg.skeleton)
        return tmpimg



if __name__ == '__main__':
    # pose_estimator = PoseEstimator()
    pose_estimator = PoseEstimatorV2()
    data_folder = '/Data/3D_pose_estimation_dataset/PIROPO/Room A/omni_1A/omni1A_test6'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS20'
    # data_folder = '/Data/3D_pose_estimation_dataset/MuCo/data/unaugmented_set/1'
    data_folder = '/Data/3D_pose_estimation_dataset/MuPoTS/data/MultiPersonTestSet/TS1'
    # data_folder = '/Data/3D_pose_estimation_dataset/RAPID'
    # save_folder = '/Data/3D_pose_estimation_dataset/demo_result'
    if __name__ == '__main__':
        for fn in sorted(os.listdir(data_folder)):
            if 'jpg' not in fn and 'png' not in fn:
                continue
            path = os.path.join(data_folder, fn)
            frame = cv2.imread(path)
            # TODO: feed bounding box of a person instead of a full image.
            frame = cv2.resize(frame, (256, 256))
            k_value = torch.tensor([3000])
            pose = pose_estimator.forward(frame, k_value)
            print(pose)
            # pickle.dump((poses, vis_kps), open(os.path.join(save_folder, fn.split('.')[0]+'.pkl'), 'wb'))
            # for pose in poses:
                # print(pose)
                # frame = pose_estimator.visualize(frame, pose)
            # vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), cfg.skeleton,
            #                          'output_pose_3d (x,y,z: camera-centered. mm.)')
            # cv2.imshow('', cv2.resize(frame, None, fx=0.25, fy=0.25))
            # cv2.waitKey()
            # cv2.imwrite(os.path.join(save_folder, fn), frame)


    # If the server recieves frames from other process use below code.
    # server = Server()
    # pose_estimation = PoseEstimator()
    # while True:
    #     frame = server.get_image()
    #     pose = pose_estimation.forward(frame)
    #     print(pose.shape)
