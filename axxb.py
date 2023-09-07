import numpy as np
import scipy.optimize as opt
import pytransform3d as p3d
from pytransform3d import transformations
import matplotlib.pyplot as plt
from pytransform3d.rotations import (
    quaternion_integrate, matrix_from_quaternion, plot_basis)
import cv2

data = np.load("calib_data_25.npz")
charuco_tf = data['charuco_tf']
tracker_tf = data['tracker_tf']

charuco_tf_inv = np.linalg.inv(charuco_tf)
tracker_tf_inv = np.linalg.inv(tracker_tf)

R, t = cv2.calibrateHandEye(charuco_tf_inv[:, :3, :3], charuco_tf_inv[:, :3, 3],
                            tracker_tf_inv[:, :3, :3], tracker_tf_inv[:, :3, 3])
X = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])))
X = np.linalg.inv(X)
print(X)

ax = None
for t in range(len(charuco_tf)):
    T = tracker_tf[t] @ X
    ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])

    T = tracker_tf[t] @ X @ charuco_tf[t]
    print(T[0, 3])
    ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])

    ax = plot_basis(ax=ax, s=0.15, R=tracker_tf[t][:3, :3], p=tracker_tf[t][:3, 3])
plt.show()
