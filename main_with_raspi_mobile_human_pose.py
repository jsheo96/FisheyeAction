# main.py
# execute our program with various options given by arguments
from pose_estimation.utils.vis import vis_3d_multiple_skeleton
import numpy as np
from pose_estimation.config import cfg
from pose_estimation.pose_estimator import PoseEstimatorV2
from pose_estimation.pose_estimator import PoseEstimatorV3
import sys
import sounddevice as sd
def audio_callback(indata, frames, time, status):
    global volume_norm
    volume_norm = np.linalg.norm(indata) * 10
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
    time.sleep(3)
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
    pose_estimatorv3 = PoseEstimatorV3(use_cuda=True)
    vis = True
    cap = VideoCapture("udp://192.168.0.3:8880")
    # Wait until cap has at least one frames in its queue.
    # while True:
    #     if cap.q.qsize() > 0:
    #         break
    while True:
        frame = cap.read()
        # cv2.imshow('', frame)
        # cv2.waitKey(1)
        # start = time.time()
        patches, k_values, sphericals = human_detector.detect(frame)
        # patches = torch.stack(patches, 0).cuda()
        if patches.shape[0] == 0:
            continue
        # patches *= 255
        # print(time.time()-start)
        """
        patches = pose_estimator.transform(patches)
        pose = pose_estimator.batch_forward(patches, k_values)

        if vis:
            patch = patches[0]
            patch = decode(patch)
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            patch = np.ascontiguousarray(patch, dtype=np.uint8)
            pose = pose[0].cpu().numpy()
            tmpimg = pose_estimator.visualize(patch, pose)
            cv2.imshow('', tmpimg)
            cv2.waitKey(1)
        """
        # cv2.imshow('patch', patches[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
        # patches = 1, 3, 256, 256 dtype=np.float32 max=255.0
        patches = pose_estimatorv3.transform(patches)
        pose = pose_estimatorv3.batch_forward(patches)

        right_elbow = pose[0,8,:,:].cpu().numpy()
        right_elbow_coord = np.unravel_index(right_elbow.argmax(), right_elbow.shape)
        right_wrist = pose[0, 10, :, :].cpu().numpy()
        right_wrist_coord = np.unravel_index(right_wrist.argmax(), right_wrist.shape)
        right_elbow_lon = sphericals[0,:,:,0][right_elbow_coord[0],right_elbow_coord[1]]
        right_wrist_lon = sphericals[0,:,:,0][right_wrist_coord[0],right_wrist_coord[1]]
        arm_direction = (right_wrist_lon - right_elbow_lon)
        print(right_elbow_lon, right_wrist_lon, arm_direction)
        if vis:
            patch = patches[0]
            patch = decode(patch)
            patch *= 255
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            patch = np.ascontiguousarray(patch, dtype=np.uint8)
            # pose = np.sum(pose[0].cpu().numpy(),axis=0)
            pose = (pose[0,10,:,:]+pose[0,8,:,:]).cpu().numpy()
            pose = np.stack((pose,)*3, -1)
            pose[:,:,:2] = 0
            pose = (pose * 255).astype(np.uint8)
            pose = cv2.resize(pose, (patch.shape[1], patch.shape[0]))
            cv2.imshow("pose", pose)
            result = cv2.addWeighted(patch, 1.0, pose, 1.0, 0.0)
            cv2.imshow('result', result)
            cv2.waitKey(1)
