import torchvision.transforms as transforms
import cv2
import numpy as np
from human_detection.fisheye_utills import FisheyeUtills as FU
from human_detection.utils.visualization import draw_xywha
from human_detection.utils.visualization import draw_dt_on_np
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

def visualize_skeleton(frame, pose, sphericals, ids, detections):
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
    line_width = frame.shape[0] // 300
    result = frame
    threshold = 0.4
    for i in range(pose.shape[0]):
        joints = []
        for j in range(pose.shape[1]):
            joint = pose[i, j, :, :].cpu().numpy()
            joint_coord = np.unravel_index(joint.argmax(), joint.shape)
            joint_lonlat = sphericals[i, :, :, :][joint_coord[0] * 4, joint_coord[1] * 4, :]
            joint_i, joint_j = img_utils.sphere2fisheye(joint_lonlat[0], joint_lonlat[1])
            joints.append((int(joint_i), int(joint_j), joint.max()))
            if joint.max() >= threshold:
                result = cv2.circle(result, (int(joint_i), int(joint_j)), color=(0, 0, 255), radius=line_width,
                                    thickness=-1)
        for j, k in skeleton:
            if min(joints[j][2], joints[k][2]) >= threshold:
                result = cv2.line(result, joints[j][:2], joints[k][:2], color=(0, 255, 0), thickness=line_width)

        # visualize ids
        # joint_lonlat = sphericals[i, 0, 0, :]
        # point_i, point_j = img_utils.sphere2fisheye(joint_lonlat[0], joint_lonlat[1])
        # cv2.putText(result, str(int(ids[i].item())), org=(int(point_i), int(point_j)), fontFace=cv2.FONT_HERSHEY_COMPLEX,
        #             fontScale=1, thickness=1, color=(0,0,255), lineType=cv2.LINE_AA)
        # x,y,w,h,a = detections[i, :5]
        # result = draw_xywha(result, x, y, w, h, a)
    draw_dt_on_np(result, detections)
    return result