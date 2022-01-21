#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

from human_detection.api import Detector
from human_detection.fisheye_utills import FisheyeUtills as FU

# should open image first
img = cv2.imread('demo/exhibition.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rapid = Detector(model_name='rapid',
                 weights_path='human_detection/weights/pL1_MWHB608_Mar11_4500.ckpt',
                 use_cuda=False)

# feed image to model
detections = rapid.detect_one(img=img,
                              visualize=False, 
                              input_size=608, 
                              conf_thres=0.7, 
                              test_aug=None)

# make fisheye utills object
fisheye_utills = FU(img=img, fov=160)
# ignore confidence values at last element
uvwha = detections[:,:5]
print('uvwha.shape:',uvwha.shape)
# returns are described below
# patches.shape     : [N, C, H, W]
# sphericals.shape: [N, 2(lon, lat), H, W]
# k_values.shape  : [N]

fig, ax = plt.subplots(figsize=(7,7))
for i, center in enumerate(detections[:,:2]):
    ax.imshow(img)
    ax.text(center[0], center[1], f'{i+1:02d}', fontsize=8, color='white', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.savefig('demo/output/original.png')

import time
now = time.time()
patches, sphericals, k_values = fisheye_utills.get_tangent_patch(uvwha,
                                                             visualize=False,
                                                             detectnet=True)
print(time.time() - now)

for i in range(patches.shape[0]):
    img = patches[i].permute(1,2,0).numpy() * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'demo/output/patches/{i:02d}.jpg', img)

