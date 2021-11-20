# TODO: get images from raspi camera and preprocess the images.

class Camera:
    def __init__(self, connection):
        self.connection = connection
        self.focal_length

    def get_frame(self):
        return frame