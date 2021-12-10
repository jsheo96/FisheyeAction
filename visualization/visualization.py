# Eunchong's implementation of 3d pose visualization
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import human_detection.fisheye_utills as FU

data = Path('./data')
pkls = sorted(list(data.glob('*.pkl')))
data_list = []
for pkl_path in pkls:
    with open(str(pkl_path), 'rb') as pkl:
        poses, vis_kps = pickle.load(pkl)
        assert len(poses) == len(vis_kps)
        for h in range(len(poses)):
            poses[h] = np.column_stack((poses[h], vis_kps[h][:, 2]))

    data_list.append(np.array(poses))

# 아웃풋 수정했으면 여기부터 실행하면 됨
# sphericals는 FU.get_tangent_patch() 의 output임

# remain person on center only
for i, d in enumerate(data_list):
    if d.size > 0:
        center_idx = np.argmin(np.mean((d[:, :, :2].mean(axis=(1)) - np.array((128, 128))) ** 2, axis=1))
        # print(i, center_idx)
        data_list[i] = d[center_idx, :, :]

# patch image pixel to virtual sphere
fu = FU()
for i, d in enumerate(data_list):
    if d.size > 0:
        def patch2sphere(row):
            lon, lat = sphericals[i][:, int(row[1]), int(row[0])]
            # print(row[-1])
            return fu.sphere2cartesian(lon=lon, lat=lat, depth=row[-1])


        data_list[i] = np.apply_along_axis(patch2sphere, 1, d)

skeleton = (
(0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2),
(2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
colors = cm.rainbow(np.linspace(0, 1, len(data_list)))

fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.view_init(elev=180 + 10, azim=270)
for i, d in enumerate(data_list):
    if d.size > 0:
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], color=colors[i])
        ax.text(d[0, 0], d[0, 1], d[0, 2], f'{i + 1:02d}')
        for sk in skeleton:
            ax.plot((d[sk[0], 0], d[sk[1], 0]), (d[sk[0], 1], d[sk[1], 1]), (d[sk[0], 2], d[sk[1], 2]), color=colors[i])

plt.show()
