from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream
import cv2
import numpy as np
class OpenpifpafPoseEstimator():
    def __init__(self):
        Predictor.loader_workers = 1
        self.predictor = Predictor(visualize_image=True, visualize_processed_image=False)

    def forward(self, patches):
        patches = [(patches[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8) for i in range(patches.shape[0])]
        iterator = iter(self.predictor.numpy_images(patches))
        preds = []
        for j,(pred, _, meta) in enumerate(iterator):
            scores = np.array([abs((anno.data[np.where(anno.data[:,2]!=0.)[0],:2]-np.array([patches[0].shape[0]/2,patches[0].shape[1]/2])).mean()) for anno in pred])
            print(scores)
            i = np.argmin(scores)
            preds.append([pred[i]]) # Assume there is only one person in the bounding box.

        return preds



if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    img = Image.open('/Data/FisheyeAction/demo/exhibition.jpg')
    img = np.array(img)
    pose_estimator = OpenpifpafPoseEstimator()
    patches = [img]#, img]
    pred = pose_estimator.forward(patches)
    print(len(pred))
