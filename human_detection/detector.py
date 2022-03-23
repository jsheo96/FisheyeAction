import cv2
from PIL import Image
from human_detection.api import Detector
from human_detection.fisheye_utills import FisheyeUtills as FU
import time
import torch
import copy
class DetectNet:
    def __init__(self, use_cuda=True):
        self.model = Detector(model_name='rapid',
                              backbone='yolov5x',
                              weights_path='/Data/FisheyeAction/human_detection/weights/rapid_pL1_yolov5x_CPHBMW608_Jan21_6000.ckpt',
                              use_cuda=use_cuda)

    def detect(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.model.detect_one(pil_img=img,
                                      visualize=False,
                                      input_size=608,#1024,
                                      conf_thres=0.4,
                                      test_aug=None,
                                      sort=True)
        if detections.shape[0] > 0:
            img_utills = FU(img, bbox_scale=1.0)
            uvwha = copy.deepcopy(detections[:,:5])
            patches, sphericals, k_values = img_utills.get_tangent_patch(uvwha,
                                                                         visualize=False,
                                                                         detectnet=True)
        else:
            patches = torch.zeros((0,3,256,256))
            k_values = torch.zeros((0))
            sphericals = torch.zeros((0,256,256,2))
        return patches, k_values, sphericals, detections