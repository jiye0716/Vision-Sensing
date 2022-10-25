from telnetlib import IP
import cv2
import numpy as np
import cv2IP

ip = cv2IP.AlphaBlend()
# 讀取圖片 load image
img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/halloween.png")
print(img1.shape)  # 圖片大小
# img1 = cv2.resize(img1, (750, 500))
# cow = img1[180:, 0:375]
# girl = img1[0:320, 375:]
fore_alpha = ip.SplitAlpha(img1)
fore = np.float32(fore_alpha[0])
# Alpha Channel
ip.ImWindow("Alpha Channel")
ip.ImShow("Alpha Channel", fore_alpha[1])
# alpha = np.float32(fore_alpha[1])/255
