import cv2
import numpy as np


def HistGraphGray(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    histGraph = np.zeros([256, 256, 3], np.uint8)
    # hist 0~255 總256個像素值出現的次數
    print(hist)
    m = max(hist)
    # print(m)
    hist = hist * 220 / m  # 縮小數值
    for h in range(256):
        n = int(hist[h])  # 整數化
        cv2.line(histGraph, (h, 255), (h, 255-n), color)
    return histGraph


img = cv2.imread(r"C:/testAI/Vision-Sensing/Imgs/back.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
color = [255, 255, 255]
histGraph1 = HistGraphGray(img1, color)
cv2.imshow("Hist Gray", histGraph1)
cv2.waitKey(0)
