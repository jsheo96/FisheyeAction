import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '..')
main_dir = os.path.join(cur_dir, 'MobileHumanPose', 'main')
sys.path.insert(0, main_dir)