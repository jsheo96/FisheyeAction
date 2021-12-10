import cv2
from PIL import Image
from api import Detector
from fisheye_utills import FisheyeUtills as FU
import numpy as np
from matplotlib import pyplot as plt
from pose_estimation.pose_estimator import PoseEstimatorV2
import time
rapid = Detector(model_name='rapid',
                 weights_path='human_detection/weights/pL1_MWHB1024_Mar11_4000.ckpt',
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
patches, k_values = process(img_path='human_detection/images/lunch.jpg', model=rapid)

pose_estimator = PoseEstimatorV2()
skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))

for patch, k_value in zip(patches, k_values):
    k_value = k_value.unsqueeze(0)
    patch = patch.permute(1,2,0).cpu().numpy()*255

    pose = pose_estimator.forward(patch,k_value)


    fig = plt.figure()
    ax = plt.axes()
    ax.set_title(f'{int(pose[0, -1]):d} mm')
    ax.imshow(patch)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for sk in skeleton:
        ax.plot((pose[sk[0], 0], pose[sk[1], 0]),
                            (pose[sk[0], 1], pose[sk[1], 1]))
    plt.show()

    patch = (patch*255).astype(np.uint8)
    frame = pose_estimator.visualize(patch, pose)
    cv2.imshow('', frame)
    cv2.waitKey()
