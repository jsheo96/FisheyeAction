import cv2
from PIL import Image
from api import Detector
from fisheye_utills import FisheyeUtills as FU
import time

rapid = Detector(model_name='rapid',
                 weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',
                 use_cuda=True)
def process(img_path, model):
    img = Image.open(img_path)
    detections = model.detect_one(pil_img=img,
                                  visualize=False,
                                  input_size=1024,
                                  conf_thres=0.4,
                                  test_aug=None)
    if detections.shape[0] > 0:
        img_utills = FU(img)
        uvwha = detections[:,:5]
        patches, sphericals, k_values = img_utills.get_tangent_patch(uvwha,
                                                                     visualize=False,
                                                                     detectnet=True)
    else:
        patches = None
    return patches, k_values
patches, k_values = process(img_path='./images/lunch.jpg', model=rapid)

for patch, k_value in zip(patches, k_values):
    cv2.imshow('',patch.permute(1,2,0).cpu().numpy()[:,:,::-1])
    print(k_value)
    cv2.waitKey()
