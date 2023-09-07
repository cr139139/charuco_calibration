import pytransform3d as p3d
from pytransform3d import rotations
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytransform3d.rotations import plot_basis

squareLength = 0.015 * 2
markerLength = 0.011 * 2
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((11, 8), squareLength, markerLength, dictionary)
board.setLegacyPattern(True)

data = np.load('intrinsics.npz')
cameraMatrix = data['camera_matrix']
distCoeffs = data['dist_coeff']

images = ["images/" + str(i + 1) + ".png" for i in range(125)]

data = np.load("calib_data_25.npz")
tracker_tf = data['tracker_tf']
ee_tf = data['robot_tf']
charuco_tf = []

for im in images:
    frame = cv2.imread(im)

    # detect markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
    # draw markers
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)
    immarkers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # draw axis
    if tvecs is not None:
        for i in range(len(tvecs)):
            cv2.drawFrameAxes(immarkers, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength / 2)

    retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board,
                                                                             cameraMatrix, distCoeffs)
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix,
                                                            distCoeffs, None, None)
    imcharuco = cv2.aruco.drawDetectedCornersCharuco(frame.copy(), charucoCorners, charucoIds)

    cv2.drawFrameAxes(imcharuco, cameraMatrix, distCoeffs, rvec, tvec, squareLength * 5)
    imcharuco_rgb = cv2.cvtColor(imcharuco, cv2.COLOR_BGR2RGB)

    rotation_matrix = cv2.Rodrigues(rvec)[0]
    board_tf = np.concatenate(
        (np.concatenate((rotation_matrix, tvec), axis=1), np.array([[0, 0, 0, 1]])))

    charuco_tf.append(board_tf)

charuco_tf = np.stack(charuco_tf)

charuco_tf_inv = np.linalg.inv(charuco_tf)
tracker_tf_inv = np.linalg.inv(tracker_tf)
ee_tf_inv = np.linalg.inv(ee_tf)

Rt, tt = cv2.calibrateHandEye(charuco_tf_inv[:, :3, :3], charuco_tf_inv[:, :3, 3],
                              tracker_tf_inv[:, :3, :3], tracker_tf_inv[:, :3, 3])
Xt = np.concatenate((np.concatenate((Rt, tt), axis=1), np.array([[0, 0, 0, 1]])))
Xt = np.linalg.inv(Xt)
print(Xt)

Re, te = cv2.calibrateHandEye(charuco_tf_inv[:, :3, :3], charuco_tf_inv[:, :3, 3],
                              ee_tf_inv[:, :3, :3], ee_tf_inv[:, :3, 3])
Xe = np.concatenate((np.concatenate((Re, te), axis=1), np.array([[0, 0, 0, 1]])))
Xe = np.linalg.inv(Xe)
print(Xe)

ax = None
display_tf_t = tracker_tf @ Xt @ charuco_tf
display_tf_e = ee_tf @ Xe @ charuco_tf

mean_tf_t = np.mean(display_tf_t[:, :3, 3], axis=0)
print(np.max(np.linalg.norm(display_tf_t[:, :3, 3] - mean_tf_t, axis=1)))

mean_tf_e = np.mean(display_tf_e[:, :3, 3], axis=0)
print(np.max(np.linalg.norm(display_tf_e[:, :3, 3] - mean_tf_e, axis=1)))

for t in range(len(charuco_tf)):
    # T = tracker_tf[t] @ Xt
    # ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])
    # T = tracker_tf[t] @ Xt @ charuco_tf[t]
    # ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])
    # ax = plot_basis(ax=ax, s=0.15, R=tracker_tf[t][:3, :3], p=tracker_tf[t][:3, 3])

    # T = ee_tf[t] @ Xe
    # ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])
    # T = ee_tf[t] @ Xe @ charuco_tf[t]
    # ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])
    # ax = plot_basis(ax=ax, s=0.15, R=ee_tf[t][:3, :3], p=ee_tf[t][:3, 3])

    T = tracker_tf[t] @ Xt @ np.linalg.inv(Xe)
    ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])
    T = tracker_tf[t] @ Xt @ np.linalg.inv(Xe) @ np.linalg.inv(ee_tf[t])
    ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])
    T = tracker_tf[t] @ Xt @ charuco_tf[t]
    ax = plot_basis(ax=ax, s=0.15, R=T[:3, :3], p=T[:3, 3])

np.savez('extrinsics.npz', opti2cam=Xt, ee2cam=Xe)

plt.show()


