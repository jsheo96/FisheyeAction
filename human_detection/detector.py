import cv2
from PIL import Image
from human_detection.api import Detector
from human_detection.fisheye_utills import FisheyeUtills as FU

class DetectNet:
    def __init__(self, use_cuda=True):
        self.model = Detector(model_name='rapid',
                         weights_path='/Data/FisheyeAction/human_detection/weights/pL1_MWHB1024_Mar11_4000.ckpt',
                         use_cuda=use_cuda)

    def detect(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        detections = self.model.detect_one(pil_img=img,
                                      visualize=False,
                                      input_size=1024,#1024,
                                      conf_thres=0.4,
                                      test_aug=None)
        if detections.shape[0] > 0:
            img_utills = FU(img)
            uvwha = detections[:,:5]
            patches, sphericals, k_values = img_utills.get_tangent_patch(uvwha,
                                                                         visualize=False,
                                                                         detectnet=True)
        else:
            patches = []
            k_values = []
        return patches, k_values