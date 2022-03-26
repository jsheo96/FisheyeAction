import torchvision.transforms as transforms
import cv2
import numpy as np
from human_detection.fisheye_utills import FisheyeUtills as FU
from human_detection.utils.visualization import draw_xywha
from human_detection.utils.visualization import draw_dt_on_np
import torch
def decode(patch):
    """
    :param patch: FloatTensor (3, H, W). Normalized image
    :return: np.array (H, W, 3). denormalized image
    """
    transform = transforms.Compose([transforms.Normalize(mean=(0.,0.,0.), std=(1/0.229, 1/0.224, 1/0.225)),
                                transforms.Normalize(mean=(-0.485, -0.456, -0.406), std=(1.,1.,1.))])
    patch = transform(patch)
    patch = patch.permute(1, 2, 0).cpu().numpy()
    return patch

def visualize_posemap(patches, pose):
    """
    Returns overlap between patch and wrist probability heatmap.
    :param patches: FloatTensor (N, 3, H, W). normalized image batch tensor
    :param pose: FlaotTensor(N, 18, h, w). stacked probability heatmaps of each joints (18)
    :return: The overlapped image of first patch image and its corresponding wrist probability heatmap
    """
    patch = patches[0]
    patch = decode(patch)
    patch *= 255
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    patch = np.ascontiguousarray(patch, dtype=np.uint8)
    pose = (pose[0, 10, :, :]).cpu().numpy()
    pose = np.stack((pose,) * 3, -1)
    pose[:, :, :2] = 0
    pose = (pose * 255).astype(np.uint8)
    pose = cv2.resize(pose, (patch.shape[1], patch.shape[0]))
    overlay = cv2.addWeighted(patch, 1.0, pose, 1.0, 0.0)
    return overlay

def visualize_skeleton(frame, pose, sphericals, detections):
    """
    Returns an image with a skeleton of one person.
    :param frame: np.array (H, W, 3)
    :param pose: torch.FloatTensor (num_person, num_joint, 64, 64)
    :param sphericals: torch.FloatTensor (num_person, 256, 256, 2)
    :param ids: torch.Tensor (num_person)
    :return: np.array (H, W, 3) BGR image. a skeleton is visualized on the returned image.
    """
    img_utils = FU(frame)
    skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12) )
    line_width = 2
    point_radius = 3
    result = frame
    threshold = 0.2
    for i in range(pose.shape[0]):
        joints = []
        for j in range(pose.shape[1]):
            pos_ind, scores = get_maximum_from_heatmap(pose[i, j, :, :].unsqueeze(0))
            joint_coord = np.unravel_index(pos_ind.item(), pose.shape[2:])
            joint_lonlat = sphericals[i, :, :, :][joint_coord[0] * 4, joint_coord[1] * 4, :]
            joint_i, joint_j = img_utils.sphere2fisheye(joint_lonlat[0], joint_lonlat[1])
            joints.append((int(joint_i), int(joint_j), scores.item()))

        for j, k in skeleton:
            if min(joints[j][2], joints[k][2]) >= threshold:
                result = cv2.line(result, joints[j][:2], joints[k][:2], color=(0, 255, 0), thickness=line_width)

        for (x, y, conf) in joints:
            if conf > threshold:
                result = cv2.circle(result, (x,y), color=(0, 0, 255), radius=point_radius,
                                    thickness=-1)
    draw_dt_on_np(result, detections)
    return result

def hierarchical_pool(heatmap):
    pool1 = torch.nn.MaxPool2d(3, 1, 1)
    pool2 = torch.nn.MaxPool2d(5, 1, 2)
    pool3 = torch.nn.MaxPool2d(7, 1, 3)
    map_size = (heatmap.shape[1]+heatmap.shape[2])/2.0
    if map_size > 300:
        maxm = pool3(heatmap[None, :, :, :])
    elif map_size > 200:
        maxm = pool2(heatmap[None, :, :, :])
    else:
        maxm = pool1(heatmap[None, :, :, :])

    return maxm
a  =0

def get_maximum_from_heatmap(heatmap):
    global a
    a+=1
    maxm = hierarchical_pool(heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    scores = heatmap.view(-1)
    if a == 81:
        print('asd')
    scores, pos_ind = scores.topk(1)
    select_ind = (scores > 0.00).nonzero()
    scores = scores[select_ind][:, 0]
    pos_ind = pos_ind[select_ind][:, 0]
    return pos_ind, scores

def openpifpaf_visualize_skeleton(frame, preds, sphericals, detections):
    img_utils = FU(frame)
    line_width = 2
    point_radius = 3
    result = frame
    threshold = 0.2
    for i in range(len(preds)):
        pred = preds[i]
        for anno in pred:
            joints = []
            for j in range(anno.data.shape[0]):
                joint_coord = min(int(anno.data[j,1]), 255), min(int(anno.data[j,0]), 255)
                joint_lonlat = sphericals[i, :, :, :][joint_coord[0], joint_coord[1], :]
                joint_i, joint_j = img_utils.sphere2fisheye(joint_lonlat[0], joint_lonlat[1])
                joints.append((int(joint_i), int(joint_j), anno.data[j,2]))
            for j, k in anno.skeleton:
                j, k = j-1, k-1
                if min(joints[j][2], joints[k][2]) >= threshold:
                    result = cv2.line(result, joints[j][:2], joints[k][:2], color=(0, 255, 0), thickness=line_width)
            for (x, y, conf) in joints:
                if conf > threshold:
                    result = cv2.circle(result, (x, y), color=(0, 0, 255), radius=point_radius,
                                        thickness=-1)
    draw_dt_on_np(result, detections)
    return result