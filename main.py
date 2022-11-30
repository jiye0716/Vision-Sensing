from telnetlib import IP
import cv2
import numpy as np
import cv2IP
ip = cv2IP.HistIP()

# 灰階
# img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/back.jpg")
# img = ip.ImBGR2Gray(img1)
# # img = ip.MonoEqualize(img) #等化
# hist = ip.CalcGrayHist(img)
# histGraph1 = ip.ShowGrayHist(hist, [255, 255, 255])
# ip.ImShow("img", histGraph1)
# cv2.waitKey(0)

# 彩色
img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/back.jpg")
ip.ImShow("img1", img1)
img = ip.ColorEqualize(img1)  # 等化
ip.ImShow("img", img)
colorRed = [0, 0, 255]
colorGreen = [0, 255, 0]
colorBlue = [255, 0, 0]
b, g, r = cv2.split(img)
# blue channel
bhist = ip.CalcGrayHist(b)
bhistGraph = ip.ShowGrayHist(bhist, colorBlue)
ip.ImShow("Hist Blue", bhistGraph)
# green channel
ghist = ip.CalcGrayHist(g)
ghistGraph = ip.ShowGrayHist(ghist, colorGreen)
ip.ImShow("Hist Green", ghistGraph)
# red channel
rhist = ip.CalcGrayHist(r)
rhistGraph = ip.ShowGrayHist(rhist, colorRed)
ip.ImShow("Hist Red", rhistGraph)
# get three channels together
histGraph2 = ip.ShowColorHist(img)
ip.ImShow("Hist Color", histGraph2)
cv2.waitKey(0)
