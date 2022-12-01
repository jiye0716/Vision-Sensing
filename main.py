from telnetlib import IP
import cv2
import numpy as np
import cv2IP
ip = cv2IP.HistIP()

# 灰階


def EQU_Gray():
    img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/is.jpg")
    img_g = ip.ImBGR2Gray(img1)
    img = ip.MonoEqualize(img_g)  # 等化
    hist_gray = ip.CalcGrayHist(img_g)
    hist_equ = ip.CalcGrayHist(img)
    histGraph_gray = ip.ShowGrayHist(hist_gray, [255, 255, 255])
    histGraph_equ = ip.ShowGrayHist(hist_equ, [255, 255, 255])
    ip.ImShow("img_gray", img_g)
    ip.ImShow("img_equ", img)
    ip.ImShow("gray_hist", histGraph_gray)
    ip.ImShow("equ_hist", histGraph_equ)
    cv2.waitKey(0)

# 彩色


def EQU_COLOR():
    img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/back.jpg")
    ip.ImShow("img1", img1)
    img = ip.ColorEqualize(img1, CType=cv2IP.ColorType.USE_YUV)  # 等化
    ip.ImShow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    #  EQU_Gray()
    EQU_COLOR()
