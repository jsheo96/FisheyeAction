# TODO: Server side image recever.
# Recieves image
from datetime import datetime
import cv2

class Server:
    def __init__(self):
        self.cap = cv2.VideoCapture("tcp://192.168.0.12:8888")

    def get_image(self):
        ret, frame = self.cap.read()
        while not ret:
            ret, frame = self.cap.read()
        return frame

    def __del__(self):
        self.cap.release()

if __name__ == '__main__':
    server = Server()
    while True:
        frame = server.get_image()
        print(frame.shape)
        cv2.imshow('server received this image', cv2.resize(frame, None, fx=0.25, fy=0.25))
        cv2.waitKey(1)
