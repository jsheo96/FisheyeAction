import sounddevice as sd
import numpy as np
from threading import Thread
class Microphone(object):
    def __init__(self):
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()
        self.max_volume_norm = 0

    def audio_callback(self, indata, frames, time, status):
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