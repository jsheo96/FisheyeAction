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

import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


if __name__ == '__main__':

    human_detector = DetectNet(use_cuda=True)
    pose_estimator = PoseEstimatorV2(use_cuda=True)
    data_folder = '/Data/3D_pose_estimation_dataset/PIROPO/Room A/omni_1A/omni1A_test6'
    data_folder = '/Data/3D_pose_estimation_dataset/CEPDOF/Edge_cases'
    vis = True
    cap = VideoCapture("udp://192.168.0.3:8880")
    # Wait until cap has at least one frames in its queue.
    while True:
        if cap.q.qsize() > 0:
            break
    while True:
        frame = cap.read()
        start = time.time()
        patches, k_values = human_detector.detect(frame)
        # patches = torch.stack(patches, 0).cuda()
        if patches.shape[0] == 0:
            continue
        patches *= 255
        patches = pose_estimator.transform(patches)
        # k_values = torch.stack(k_values, 0).unsqueeze(-1).cuda()
        pose = pose_estimator.batch_forward(patches, k_values)
        print(time.time()-start)
        if vis:
            patch = patches[0]
            patch = decode(patch)
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            patch = np.ascontiguousarray(patch, dtype=np.uint8)
            pose = pose[0].cpu().numpy()
            tmpimg = pose_estimator.visualize(patch, pose)
            cv2.imshow('', tmpimg)
            cv2.waitKey(1)
