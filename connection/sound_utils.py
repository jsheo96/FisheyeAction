import sounddevice as sd
import numpy as np
from threading import Thread
class Microphone(object):
    def __init__(self):
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()
        self.max_volume_norm = 0

    def audio_callback(self, indata, frames, time, status):
        """
        Callback for stream of sounddevice. Called when sound input data is updated.
        maximum_volumne_norm is updated every time this function is called.
        maximum_volume_norm is used for detecting triggers, e.g. clapping, snapping.
        :param indata: incomming sound data from microphone device.
        :param frames: Not used
        :param time: Not used
        :param status: Not used
        :return: None
        """
        # global volume_norm, max_volume_norm
        volume_norm = np.linalg.norm(indata) * 10
        if self.max_volume_norm < volume_norm:
            self.max_volume_norm = volume_norm

    def __del__(self):
        self.stream.close()

if __name__ == '__main__':
    import time
    audio = Microphone()
    while True:
        time.sleep(1)
        print('s')