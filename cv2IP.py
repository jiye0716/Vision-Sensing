import cv2
import numpy as np
import enum


class BaseIP(object):
    # 影像讀取、儲存與顯示函數
    @staticmethod  # 影像儲存
    def ImRead(filename):
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    @staticmethod  # 顯示影像
    def ImWrite(filename, img):
        return cv2.imwrite(filename, img)

    @staticmethod  # cv.destroyAllWindows()
    def ImShow(winname, img):
        cv2.imshow(winname, img)

    @staticmethod  # 創建顯示視窗
    def ImWindow(winname):
        cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)


class AlphaBlend(BaseIP):
    @staticmethod
    def SplitAlpha(SrcImg):
        b, g, r, a = cv2.split(SrcImg)
        fore = cv2.merge([b, g, r])
        alpha = cv2.merge([a, a, a])
        return fore, alpha

    @staticmethod
    def DoBlending(Foreground, Background, Alpha):
        fore = cv2.multiply(Foreground, Alpha)
        back = cv2.multiply(Background, 1.0-Alpha)
        out = cv2.add(fore, back)
        return out


class ColorType(enum.IntEnum):
    USE_RGB = 1
    USE_HSV = 2
    USE_YUV = 3


class HistIP(BaseIP):
    @staticmethod
    def ImBGR2Gray(SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def ImBGRA2BGR(SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def CalcGrayHist(SrcGray):
        #(  影像 ,通道,遮罩,區間數量, 數值範圍 )
        return cv2.calcHist([SrcGray], [0], None, [256], [0.0, 255.0])

    @staticmethod
    def ShowGrayHist(hist, color):
        histGraph = np.zeros([256, 256, 3], np.uint8)
        m = max(hist)
        hist = hist * 220 / m
        for h in range(256):
            n = int(hist[h])
            cv2.line(histGraph, (h, 255), (h, 255-n), color)
        return histGraph

    @staticmethod
    def CalcColorHist(SrcColor):
        SrcColor1 = cv2.calcHist(SrcColor, [0], None, [256], [0, 256])
        SrcColor2 = cv2.calcHist(SrcColor, [1], None, [256], [0, 256])
        SrcColor3 = cv2.calcHist(SrcColor, [2], None, [256], [0, 256])
        return SrcColor1, SrcColor2, SrcColor3

    @staticmethod
    def ShowColorHist(image):
        histGraph = np.zeros([256, 256, 3], np.uint8)
        colorBlue = [255, 0, 0]
        colorGreen = [0, 255, 0]
        colorRed = [0, 0, 255]
        b, g, r = cv2.split(image)
        bhist = cv2.calcHist([b], [0], None, [256], [0.0, 255.0])
        ghist = cv2.calcHist([g], [0], None, [256], [0.0, 255.0])
        rhist = cv2.calcHist([r], [0], None, [256], [0.0, 255.0])
        bm = max(bhist)
        gm = max(ghist)
        rm = max(rhist)
        bhist = bhist * 220 / bm
        rhist = rhist * 220 / rm
        ghist = ghist * 220 / gm
        for h in range(256):
            bn = int(bhist[h])
            gn = int(ghist[h])
            rn = int(rhist[h])
            if h != 0:
                cv2.line(histGraph, (h-1, 255-bStart), (h, 255-bn), colorBlue)
                cv2.line(histGraph, (h-1, 255-gStart), (h, 255-gn), colorGreen)
                cv2.line(histGraph, (h-1, 255-rStart), (h, 255-rn), colorRed)
            bStart = bn
            gStart = gn
            rStart = rn
        return histGraph

    @staticmethod
    def MonoEqualize(SrcGray):
        SrcGray = cv2.equalizeHist(SrcGray)
        return SrcGray

    class ColorType(enum.IntEnum):
        USE_RGB = 1
        USE_HSV = 2
        USE_YUV = 3

    @staticmethod
    def ColorEqualize(SrcColor, CType=ColorType.USE_HSV):
        hsv = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v2 = cv2.equalizeHist(v)
        s2 = cv2.equalizeHist(s)
        img2 = cv2.merge([h, s2, v2])
        rgb = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
        return rgb

    @staticmethod
    def HistMatching(SrcImg, RefImg, CType=ColorType.USE_HSV):
