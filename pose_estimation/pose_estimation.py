# TODO: Do 3D Human Pose Estimation from given images.

from connection.server import Server
class PoseEstimation:
    def __init__(self):
        # TODO: initialize models
        pass

    def forward(self, image):
        # TODO: process image to 3d skeleton
        pass

if __name__ == '__main__':
    server = Server()
    pose_estimation = PoseEstimation()
    while True:
        frame = server.get_image()
        pose = pose_estimation.forward(frame)
        print(pose)
