import time
import numpy as np
import requests
class ArmClapTrigger(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.bulb_left = False
        self.bulb_right = False
        self.token = cfg['token']
        self.raspi = cfg['raspi']
        self.raspi_ip = cfg['raspi_ip']
        self.raspi_port = cfg['raspi_port']
        self.start = time.time()

    def run(self, pose, sphericals, mic):
        """
        Gets skeleton information and triggers an action (toggle lights) if conditions(sound volume > threshold) are fulfilled.
        Detects trigger condition of the first person detected(0).
        Arm direction is obtained from wrist and elbow.
        Trigger condition is whenever the inputted sound volume is above a threshold.
        :param pose: np.array (B, num_joints, 64, 64)
        :param sphericals: np.array (B, 256, 256, 2)
        :param mic: connection.sound_utils.Microphone.
        :return: None
        """
        right_elbow = pose[0, 8, :, :].cpu().numpy()
        right_elbow_coord = np.unravel_index(right_elbow.argmax(), right_elbow.shape)
        right_wrist = pose[0, 10, :, :].cpu().numpy()
        right_wrist_coord = np.unravel_index(right_wrist.argmax(), right_wrist.shape)
        right_elbow_lon = sphericals[0, :, :, 0][right_elbow_coord[0] * 4, right_elbow_coord[1] * 4]
        right_elbow_lat = sphericals[0, :, :, 1][right_elbow_coord[0] * 4, right_elbow_coord[1] * 4]
        right_wrist_lon = sphericals[0, :, :, 0][right_wrist_coord[0] * 4, right_wrist_coord[1] * 4]
        right_wrist_lat = sphericals[0, :, :, 1][right_wrist_coord[0] * 4, right_wrist_coord[1] * 4]

        arm_direction = (right_wrist_lon - right_elbow_lon)
        if time.time() - self.start < 0.5:
            mic.max_volume_norm = 0
        if mic.max_volume_norm >= 10 and time.time() - self.start >= 0.5:
            if arm_direction >= 0:
                print('RIGHT triggered!!!')
                self.bulb_right = not self.bulb_right
                trigger = 'on' if self.bulb_right else 'off'
                url = f"http://{self.raspi_ip}:{self.raspi_port}/api/services/light/turn_{trigger}"
                headers = {"Authorization": "Bearer {}".format(self.token)}
                data = {"entity_id": "light.tall"}
                if self.raspi:
                    response = requests.post(url, headers=headers, json=data)
                    print(response.text)
                self.start = time.time()
            else:
                print('LEFT triggered!!!')
                self.bulb_left = not self.bulb_left
                trigger = 'on' if self.bulb_left else 'off'
                url = f"http://{self.raspi_ip}:{self.raspi_port}/api/services/light/turn_{trigger}"
                headers = {"Authorization": "Bearer {}".format(self.token)}
                data = {"entity_id": "light.short"}
                if self.raspi:
                    response = requests.post(url, headers=headers, json=data)
                    print(response.text)
                self.start = time.time()
            mic.max_volume_norm = 0

