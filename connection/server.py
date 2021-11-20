# TODO: Server side image recever.
# Recieves image
from datetime import datetime
import imagezmq
import cv2

class Server:
    def __init__(self):
        self.imageHub = imagezmq.ImageHub()
        self.lastActive = {}

    def get_image(self):
        (rpiName, frame) = self.imageHub.recv_image()
        self.imageHub.send_reply(b'OK')
        if rpiName not in self.lastActive.keys():
            print("[INFO] receiving data from {}...".format(rpiName))
        self.lastActive[rpiName] = datetime.now()
        return frame

if __name__ == '__main__':
    server = Server()
    while True:
        frame = server.get_image()
        cv2.imshow('server received this image', frame)
        cv2.waitKey(1)
