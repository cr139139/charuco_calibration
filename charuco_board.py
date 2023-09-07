import cv2
from cv2 import aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
# aruco_dict.bytesList = aruco_dict.bytesList[:44, :, :][::-1, :, :]
board = aruco.CharucoBoard((11, 8), 0.015, 0.011, aruco_dict)
board.setLegacyPattern(True)
img = aruco.CharucoBoard.generateImage(board, (1000, 1000), marginSize=0)
# cv2.imwrite('image.png', imboard)
cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

