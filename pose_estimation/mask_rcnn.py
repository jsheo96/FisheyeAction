from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
import cv2
from pose_estimation.predictor import COCODemo


# COCODemo Wrapper
class MaskRCNN:
    def __init__(self):
        config_file = "/Data/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        coco_demo = COCODemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
        )
        self.coco_demo = coco_demo

    def detect(self, image):
        predictions = self.coco_demo.compute_prediction(image)
        predictions = self.coco_demo.select_top_predictions(predictions)
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        return labels, boxes

def get_mask_rcnn():
    return MaskRCNN()