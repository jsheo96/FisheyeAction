import os
import cv2
import queue
import threading

class VideoCapture:
    def __init__(self, name="udp://127.0.0.1:8880"):
        """
        Initializes VideoCaptrue instance.
        When UDP connection is not completed, retries to connect to camera.
        Queue structure is used for keeping frames (FIFO)
        :param name:
        """

        while True:
            self.cap = cv2.VideoCapture(name)
            if self.cap.grab():
                break
            print("Retrying camera connection")
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        """
        Reads a frame from a connected camera.
        Keeps only the most recent one and discards old frames.
        :return: None
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        """
        Returns the most recent frame obtained from camera.
        :return: np.array with shape (1024, 1024, 3). BGR image.
        """
        return self.q.get()

class FolderCapture(object):
    def __init__(self, folder='/Data/3D_pose_estimation_dataset/PIROPO/Room A/omni_1A/omni1A_test1'):
        self.folder = folder
        self.image_paths = []
        for fn in sorted(os.listdir(self.folder)):
            if os.path.splitext(fn)[1][1:] not in ['jpg','png']:
                continue
            path = os.path.join(self.folder, fn)
            self.image_paths.append(path)
        self.i = 0

    def read(self):
        """
        :return: (H, W, 3) np.array. BGR image.
        """
        img = cv2.imread(self.image_paths[self.i])
        img = self.crop_and_resize(img)
        self.i += 1
        return img

    def crop_and_resize(self, img):
        """
        Crop center part of image and resize to square shape.
        :param img: np.array (H, W, 3) BGR image.
        :return: np.array (H, W, 3) BGR image. H == W
        """
        h,w,_ = img.shape
        if h < w:
            d = (w - h)//2
            img = img[:, d:w-d, :]
        else:
            d = (h - w) // 2
            img = img[d:h - d, :, :]
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        return img