#!/usr/bin/env python
import rospy
import tf2_ros
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import pytransform3d as p3d
from pytransform3d import rotations
import pyrealsense2 as rs
import numpy as np
import cv2

rospy.init_node('tf2_listener')
tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

def tf2_list(tf2):
    return [tf2.transform.translation.x,
            tf2.transform.translation.y,
            tf2.transform.translation.z,
            tf2.transform.rotation.w,
            tf2.transform.rotation.x,
            tf2.transform.rotation.y,
            tf2.transform.rotation.z]

current_joint = np.zeros(7)


def callback(reading):
    global current_joint
    current_joint = np.array(reading.position)


pub = rospy.Publisher('/iiwa/PositionController/command', Float64MultiArray, queue_size=100)
sub = rospy.Subscriber("/iiwa/joint_states", JointState, callback)
ros_msg = Float64MultiArray()

calib_joint = np.load("joint_calib_poses.npz")['q']
joint_index = 0

squareLength = 0.015 * 2
markerLength = 0.011 * 2
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((11, 8), squareLength, markerLength, dictionary)
board.setLegacyPattern(True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg = pipeline.start(config)

profile = cfg.get_stream(rs.stream.color, 0)
intr = profile.as_video_stream_profile().get_intrinsics()
cameraMatrix = np.array([[intr.fx, 0, intr.ppx],
                         [0, intr.fy, intr.ppy],
                         [0, 0, 1]])
distCoeffs = np.zeros(4)

data_camera = []
data_optitrack = []
data_ee = []
index = 0
print('loop started')
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # detect markers
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        for corner in corners:
            cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
        # draw markers
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)
        immarkers = cv2.aruco.drawDetectedMarkers(color_image.copy(), corners, ids)
        # draw axis
        if tvecs is not None:
            for i in range(len(tvecs)):
                cv2.drawFrameAxes(immarkers, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength / 2)

        if corners is not None and ids is not None:
            if len(corners) > 8:
                retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board,
                                                                                         cameraMatrix, distCoeffs)
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix,
                                                                        distCoeffs, None, None)
                imcharuco = cv2.aruco.drawDetectedCornersCharuco(color_image.copy(), charucoCorners, charucoIds)
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(imcharuco, cameraMatrix, distCoeffs, rvec, tvec, squareLength * 5)
                    imcharuco_rgb = cv2.cvtColor(imcharuco, cv2.COLOR_BGR2RGB)

                    rotation_matrix = cv2.Rodrigues(rvec)[0]
                    charuco_tf = np.concatenate(
                        (np.concatenate((rotation_matrix, tvec), axis=1), np.array([[0, 0, 0, 1]])))
                    cv2.imshow('Stream', imcharuco)
        try:
            tracker_tf = tfBuffer.lookup_transform('world', 'kuka_ee', rospy.Time(0))
            T = np.array(tf2_list(tracker_tf))
            R = p3d.rotations.matrix_from_quaternion(T[3:])
            tracker_tf = np.concatenate((np.concatenate((R, T[:3][:, np.newaxis]), axis=1), np.array([[0, 0, 0, 1]])))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue

        try:
            ee_tf = tfBuffer.lookup_transform('world', 'iiwa_link_ee', rospy.Time(0))
            T = np.array(tf2_list(ee_tf))
            R = p3d.rotations.matrix_from_quaternion(T[3:])
            ee_tf = np.concatenate((np.concatenate((R, T[:3][:, np.newaxis]), axis=1), np.array([[0, 0, 0, 1]])))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue

        target_joint = np.arctan2(np.sin(calib_joint[joint_index]), np.cos(calib_joint[joint_index]))
        ros_msg.data = 0.1 * (target_joint - current_joint) + current_joint
        pub.publish(ros_msg)
        print(np.linalg.norm(target_joint - current_joint))

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            index += 1
            print("save {0:d} transformation matrix!".format(index))
            print(charuco_tf)
            print(tracker_tf)
            print(ee_tf)
            data_camera.append(charuco_tf)
            data_optitrack.append(tracker_tf)
            data_ee.append(ee_tf)
            cv2.imwrite('images/'+str(index)+'.png', color_image)
        elif key == ord('a'):
            joint_index -= 1
            joint_index %= calib_joint.shape[0]
        elif key == ord('d'):
            joint_index += 1
            joint_index %= calib_joint.shape[0]
finally:
    pipeline.stop()
    np.savez("calib_data_25.npz",
             charuco_tf=np.stack(data_camera),
             tracker_tf=np.stack(data_optitrack),
             robot_tf=np.stack(data_ee))