# TODO: read PIROPO and send images to sender like a client of imagezmq
import cv2
import os
import imagezmq
import argparse
import socket
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
    help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
    args["server_ip"]))
rpiName = socket.gethostname()
piropo_folder = '/Data/3D_pose_estimatinon_dataset/Room A/omni_1A/omni1A_test6'
if __name__ == '__main__':
    for fn in sorted(os.listdir(piropo_folder)):
        if 'jpg' not in fn:
            continue
        path = os.path.join(piropo_folder, fn)
        frame = cv2.imread(path)
        cv2.imshow('client send this image', frame)
        cv2.waitKey(100)
        sender.send_image(rpiName, frame)