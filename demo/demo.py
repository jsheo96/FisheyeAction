import sys
sys.path.append("..")

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

from human_detection.api import Detector
from human_detection.fisheye_utills import FisheyeUtills as FU
from pose_estimation.pose_estimator import PoseEstimatorV2 as PE

# should open image first
img = Image.open('./exhibition.jpg')  

rapid = Detector(model_name='rapid',
                 weights_path='../human_detection/weights/pL1_MWHB1024_Mar11_4000.ckpt',
                 use_cuda=False)

# feed image to model
detections = rapid.detect_one(pil_img=img,
                              visualize=False, 
                              input_size=1024, 
                              conf_thres=0.7, 
                              test_aug=None)
# convert fisheye images
if detections.shape[0] > 0:
    pass
else:
    sys.exit()
    
# make fisheye utills object
fisheye_utills = FU(img=img, fov=160)
# ignore confidence values at last element
uvwha = detections[:,:5]
# returns are described below
# patch.shape     : [N, C, H, W]
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

plt.savefig('./output/original.png')

patches, sphericals, k_values = fisheye_utills.get_tangent_patch(uvwha,
                                                             visualize=False,
                                                             detectnet=True)


pose_estimator = PE()
poses = []
for i, patch in enumerate(patches):
    p = patch.permute(1,2,0).numpy() * 255
    poses.append(pose_estimator.forward(p, k_values[i]))
    
# patch image pixel to virtual sphere
shperical_poses = []
for i, p in enumerate(poses):
    if p.size > 0:
        def patch2sphere(row):
            lon, lat = sphericals[i][:, int(row[1]), int(row[0])]
            return fisheye_utills.sphere2cartesian(lon=lon, lat=lat, depth=row[-1])
        shperical_poses.append(np.apply_along_axis(patch2sphere, 1, p))
        
skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
colors = cm.rainbow(np.linspace(0, 1, len(shperical_poses)))

fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection='3d')
ax.view_init(elev=50, azim=-90+0)
ax.plot((0,0),(0,0),(-250,2500), color='black', linewidth=2, alpha=0.5)
for i, p in enumerate(shperical_poses):
    if p.size > 0:
        ax.scatter(p[:,0], p[:,1], p[:,2], s=1, color=colors[i], alpha=0.5)
        ax.text(p[0,0], p[0,1], p[0,2], f'{i+1:02d}', fontsize=6, fontweight='bold')
        ax.set_aspect('auto')
        ax.set_ylim((-4000, 4000))
        ax.set_ylim((4000, -4000))
        ax.set_zlim((3000, -5000))
        for sk in skeleton:
            ax.plot((p[sk[0],0], p[sk[1],0]),
                    (p[sk[0],1], p[sk[1],1]),
                    (p[sk[0],2], p[sk[1],2]),
                    color=colors[i], linewidth=2, alpha=0.3)

plt.savefig('./output/3d_plot.png')

ncols = int(np.round(np.sqrt(len(patches))))
nrows = len(patches)//ncols + 1

fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7,7))
for row in range(nrows):
    for col in range(ncols):
        idx = row * ncols + col
        if idx < len(patches):
            axes[row, col].set_title(f'{idx+1:02d}. {poses[idx][0,-1]/1000:.1f}m')
            axes[row, col].imshow(patches[idx].permute(1,2,0))
            # axes[row, col].scatter(poses[idx][:,0], poses[idx][:,1], color=colors[idx], alpha=0.3)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_xticklabels([])
            axes[row, col].set_yticklabels([])
            for sk in skeleton:
                axes[row, col].plot((poses[idx][sk[0],0], poses[idx][sk[1],0]),
                                    (poses[idx][sk[0],1], poses[idx][sk[1],1]),
                                    color=colors[idx], linewidth=2, alpha=0.4)
            
        else:
            axes[row, col].remove()
            
plt.tight_layout()
plt.savefig('./output/patches.png')