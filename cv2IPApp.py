from telnetlib import IP
import cv2
import numpy as np
import cv2IP

ip = cv2IP.HistIP()
# 讀取圖片 load image
img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/Angelina-Jolie.png")
img1_BGR = ip.ImBGRA2BGR(img1)
img1_GRAY = ip.ImBGR2Gray(img1_BGR)
Hist = ip.CalcGrayHist(img1_GRAY)
ip.ShowGrayHist(Hist, [256, 0, 0])
# img1 = cv2.resize(img1, (750, 750))
# back = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/teacher.jpg")
# back = cv2.resize(back, (750, 750))
# fore, alpha = ip.SplitAlpha(img1)
# fore = np.float32(fore)
# alpha = np.float32(alpha)/255
# ImDim = np.shape(fore)

# if (ImDim[0] != back.shape[0] or ImDim[1] != back.shape[1]):
#     back = cv2.resize(back, (ImDim[1], ImDim[0]))
# back = np.float32(back)
# out = ip.DoBlending(fore, back, alpha)
# out = np.uint8(out)
# ip.ImWindow("AlphaBlending Result")
#ip.ImShow("AlphaBlending Result", img1_GRAY)
# ip.ImShow("test", img1)
