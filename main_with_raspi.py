from pose_estimation.pose_estimator import PoseEstimatorV3
from connection.sound_utils import Microphone
from connection.video_utils import VideoCapture
import cv2
from human_detection.detector import DetectNet
from visualization.vis_utils import visualize_skeleton
from connection.video_utils import FolderCapture
import yaml
from action_recognition.triggers import ArmClapTrigger
import os
import sys
import time
sys.path.insert(0, 'human_detection')
if __name__ == '__main__':
    with open('configs/config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if cfg['raspi']:
        os.system('ssh -X pi@192.168.0.31 sh /home/pi/video_on_.sh &')
    human_detector = DetectNet(use_cuda=True)
    pose_estimator = PoseEstimatorV3(use_cuda=True)
    cap = VideoCapture() if cfg['raspi'] else FolderCapture()
    mic = Microphone()
    record = False
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter('out.mp4', fourcc, 15.0, (1024,1024))
        n = 0
    trigger = ArmClapTrigger(cfg)
    while True:
        frame = cap.read()
        patches, k_values, sphericals, detections = human_detector.detect(frame)
        if patches.shape[0] == 0:
            cv2.imshow("result", frame)
            cv2.waitKey(1)
            continue
        patches = pose_estimator.transform(patches)
        pose = pose_estimator.batch_forward(patches)
        trigger.run(pose, sphericals, mic)
        if cfg['vis']:
            result = visualize_skeleton(frame, pose, sphericals, detections)
            cv2.imshow("result", result)
            cv2.waitKey(1)
            if record:
                writer.write(result)
                n += 1
                if n > 1000:
                    break
    writer.release()
