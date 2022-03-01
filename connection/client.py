# TODO: Client side image sender. This script should be running on raspberry pi before execution of our program
from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", default='192.168.0.25',
    help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())
sender = imagezmq.ImageSender(connect_to="tcp://{}:8888".format(
    args["server_ip"]))
rpiName = socket.gethostname()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
while True:
    frame = vs.read()
    sender.send_image(rpiName, frame)
