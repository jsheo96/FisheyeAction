# main.py
# execute our program with various options given by arguments
from pose_estimation.utils.vis import vis_3d_multiple_skeleton
import numpy as np
from pose_estimation.config import cfg
from pose_estimation.pose_estimator import PoseEstimatorV2
from pose_estimation.pose_estimator import PoseEstimatorV3
import sys
import requests
import sounddevice as sd
from human_detection.fisheye_utills import FisheyeUtills as FU
def audio_callback(indata, frames, time, status):
    global volume_norm, max_volume_norm
    volume_norm = np.linalg.norm(indata) * 10
    print('|'*int(volume_norm))
    if max_volume_norm < volume_norm:
        max_volume_norm = volume_norm


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
    pose_estimatorv3 = PoseEstimatorV3(use_cuda=True)
    vis = True
    cap = VideoCapture("udp://192.168.0.3:8880")
    # Wait until cap has at least one frames in its queue.
    # while True:
    #     if cap.q.qsize() > 0:
    #         break
    stream = sd.InputStream(callback=audio_callback)

    token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI4NjBkOTdhOGY1NmM0OTY5OGVkMDZkYjg4Y2Q1ZjBmZiIsImlhdCI6MTY0NTg5NTY2MiwiZXhwIjoxOTYxMjU1NjYyfQ.L7mvKMyrRcGADZD3-SFb8-USg8HywbHy_Cq32tUy0NQ'
    start = time.time()
    bulb_left = False
    bulb_right = False
    max_volume_norm = 0
    time.sleep(3)
    skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12) )
    record = False
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter('out.mp4', fourcc, 15.0, (1024,1024))
        n = 0
    with stream:
        while True:
            frame = cap.read()
            # cv2.imshow('', frame)
            # cv2.waitKey(1)
            # start = time.time()
            img_utils = FU(frame)
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

            right_elbow = pose[0, 8, :,:].cpu().numpy()
            right_elbow_coord = np.unravel_index(right_elbow.argmax(), right_elbow.shape)
            right_wrist = pose[0, 10, :, :].cpu().numpy()
            right_wrist_coord = np.unravel_index(right_wrist.argmax(), right_wrist.shape)
            right_elbow_lon = sphericals[0,:,:,0][right_elbow_coord[0]*4,right_elbow_coord[1]*4]
            right_elbow_lat = sphericals[0, :, :, 1][right_elbow_coord[0]*4, right_elbow_coord[1]*4]
            right_wrist_lon = sphericals[0,:,:,0][right_wrist_coord[0]*4,right_wrist_coord[1]*4]
            right_wrist_lat = sphericals[0, :, :, 1][right_wrist_coord[0]*4, right_wrist_coord[1]*4]

            arm_direction = (right_wrist_lon - right_elbow_lon)
            if time.time() - start < 0.5:
                max_volume_norm = 0
            if max_volume_norm >= 10 and time.time() - start >= 0.5:
                if arm_direction >= 0:
                    print('RIGHT triggered!!!')
                    bulb_right = not bulb_right
                    trigger = 'on' if bulb_right else 'off'
                    url = "http://192.168.0.32:8123/api/services/light/turn_{}".format(trigger)
                    headers = {"Authorization": "Bearer {}".format(token)}
                    data = {"entity_id": "light.tall"}
                    response = requests.post(url, headers=headers, json=data)
                    print(response.text)
                    start = time.time()
                else:
                    print('LEFT triggered!!!')
                    bulb_left = not bulb_left
                    trigger = 'on' if bulb_left else 'off'
                    url = "http://192.168.0.32:8123/api/services/light/turn_{}".format(trigger)
                    headers = {"Authorization": "Bearer {}".format(token)}
                    data = {"entity_id": "light.short"}
                    response = requests.post(url, headers=headers, json=data)
                    print(response.text)
                    start = time.time()
                max_volume_norm = 0
            if vis:
                # right_elbow_i, right_elbow_j = img_utils.sphere2fisheye(right_elbow_lon, right_elbow_lat)
                # right_wrist_i, right_wrist_j = img_utils.sphere2fisheye(right_wrist_lon, right_wrist_lat)
                result = frame
                # result = cv2.circle(frame, (int(right_elbow_i), int(right_elbow_j)), color=(0,0,255), radius=3, thickness=-1)
                # result = cv2.circle(result, (int(right_wrist_i), int(right_wrist_j)), color=(0,0,255), radius=3, thickness=-1)
                joints = []
                threshold = 0.4
                for i in range(pose.shape[1]):
                    joint = pose[0, i, :, :].cpu().numpy()
                    joint_coord = np.unravel_index(joint.argmax(), joint.shape)
                    joint_lonlat = sphericals[0, :, :, :][joint_coord[0] * 4, joint_coord[1] * 4, :]
                    joint_i, joint_j = img_utils.sphere2fisheye(joint_lonlat[0], joint_lonlat[1])
                    joints.append((int(joint_i),int(joint_j), joint.max()))
                    if joint.max() >= threshold:
                        result = cv2.circle(result, (int(joint_i), int(joint_j)), color=(0, 0, 255), radius=3,
                                            thickness=-1)
                for i,j in skeleton:
                    if joints[i][2] >= threshold and joints[j][2] >= threshold:
                        result = cv2.line(result, joints[i][:2], joints[j][:2], color=(0,255,0), thickness=1)

                cv2.imshow("result", result)

                patch = patches[0]
                patch = decode(patch)
                patch *= 255
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                patch = np.ascontiguousarray(patch, dtype=np.uint8)
                # pose = np.sum(pose[0].cpu().numpy(),axis=0)
                pose = (pose[0,10,:,:]).cpu().numpy()
                pose = np.stack((pose,)*3, -1)
                pose[:,:,:2] = 0
                pose = (pose * 255).astype(np.uint8)
                pose = cv2.resize(pose, (patch.shape[1], patch.shape[0]))
                coord = np.unravel_index(pose.argmax(), pose.shape)
                # cv2.imshow('patch_pose', cv2.circle(patch, (right_wrist_coord[1]*4, right_wrist_coord[0]*4), radius=3, color=(0,0,255),thickness=-1))
                # cv2.imshow("pose", pose)
                # overlay = cv2.addWeighted(patch, 1.0, pose, 1.0, 0.0)
                # cv2.imshow('overlay',overlay)
                cv2.waitKey(1)
                if record:
                    writer.write(result)
                    n += 1
                    if n > 1000:
                        break
        # writer.release()

