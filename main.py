from telnetlib import IP
import cv2
import numpy as np
import cv2IP
ip = cv2IP.HistIP()

# 灰階


def EQU_Gray():
    img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/back.jpg")
    img = ip.ImBGR2Gray(img1)
    # img = ip.MonoEqualize(img)  # 等化
    hist = ip.CalcGrayHist(img)
    histGraph1 = ip.ShowGrayHist(hist, [255, 255, 255])
    ip.ImShow("img", img)
    ip.ImShow("gray_hist", histGraph1)
    cv2.waitKey(0)

# 彩色


def EQU_COLOR():
    img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/back.jpg")
    ip.ImShow("img1", img1)
    img = ip.ColorEqualize(img1, CType=cv2IP.ColorType.USE_YUV)  # 等化
    ip.ImShow("img", img)

    b, g, r = cv2.split(img)
    # blue channel
    bhist = ip.CalcGrayHist(b)
    bhistGraph = ip.ShowGrayHist(bhist, [255, 0, 0])
    ip.ImShow("Hist Blue", bhistGraph)
    # green channel
    ghist = ip.CalcGrayHist(g)
    ghistGraph = ip.ShowGrayHist(ghist, [0, 255, 0])
    ip.ImShow("Hist Green", ghistGraph)
    # red channel
    rhist = ip.CalcGrayHist(r)
    rhistGraph = ip.ShowGrayHist(rhist, [0, 0, 255])
    ip.ImShow("Hist Red", rhistGraph)
    # get three channels together
    bh, gh, rh = ip.CalcColorHist(img)
    histGraph2 = ip.ShowColorHist(img, bh, gh, rh)
    ip.ImShow("Hist Color", histGraph2)
    cv2.waitKey(0)


if __name__ == '__main__':
    EQU_Gray()
    #  EQU_COLOR()
