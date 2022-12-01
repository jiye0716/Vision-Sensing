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
    def ShowColorHist(image, bhist, ghist, rhist):
        histGraph = np.zeros([256, 256, 3], np.uint8)
        b, g, r = cv2.split(image)
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
                cv2.line(histGraph, (h-1, 255-bStart),
                         (h, 255-bn), [255, 0, 0])
                cv2.line(histGraph, (h-1, 255-gStart),
                         (h, 255-gn), [0, 255, 0])
                cv2.line(histGraph, (h-1, 255-rStart),
                         (h, 255-rn), [0, 0, 255])
            bStart = bn
            gStart = gn
            rStart = rn
        return histGraph

    @staticmethod
    def MonoEqualize(SrcGray):
        EqualizeGray = cv2.equalizeHist(SrcGray)
        return EqualizeGray

    @staticmethod
    def ColorEqualize(SrcColor, CType=ColorType.USE_HSV):
        if CType == ColorType.USE_RGB:
            b, g, r = cv2.split(SrcColor)
            EqualizeBlue = cv2.equalizeHist(b)
            EqualizeGreen = cv2.equalizeHist(g)
            EqualizeRed = cv2.equalizeHist(r)

            # blue channel
            bhist = HistIP.CalcGrayHist(b)
            bhistGraph = HistIP.ShowGrayHist(bhist, [255, 0, 0])
            HistIP.ImShow("Hist Blue", bhistGraph)
            # green channel
            ghist = HistIP.CalcGrayHist(g)
            ghistGraph = HistIP.ShowGrayHist(ghist, [0, 255, 0])
            HistIP.ImShow("Hist Green", ghistGraph)
            # red channel
            rhist = HistIP.CalcGrayHist(r)
            rhistGraph = HistIP.ShowGrayHist(rhist, [0, 0, 255])
            HistIP.ImShow("Hist Red", rhistGraph)

            # get three channels together

            EqualizeColor = cv2.merge(
                [EqualizeBlue, EqualizeGreen, EqualizeRed])
            bh, gh, rh = HistIP.CalcColorHist(EqualizeColor)
            histGraph2 = HistIP.ShowColorHist(EqualizeColor, bh, gh, rh)
            HistIP.ImShow("Hist Color", histGraph2)
        elif CType == ColorType.USE_HSV:
            hsv = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2HSV)
            Hist_V = cv2.calcHist(hsv, [2], None, [256], [0, 256])
            histGraph_V = HistIP.ShowGrayHist(Hist_V, [255, 255, 255])
            HistIP.ImShow("v", histGraph_V)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # Value明度
            Hist_vequ = cv2.calcHist(hsv, [2], None, [256], [0, 256])
            histGraph_vequ = HistIP.ShowGrayHist(Hist_vequ, [255, 255, 255])
            HistIP.ImShow("v_equ", histGraph_vequ)
            EqualizeColor = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif CType == ColorType.USE_YUV:
            yuv = cv2.cvtColor(SrcColor, cv2.COLOR_BGR2YUV)
            Hist_y = cv2.calcHist(yuv, [0], None, [256], [0, 256])
            histGraph_y = HistIP.ShowGrayHist(Hist_y, [255, 255, 255])
            HistIP.ImShow("y", histGraph_y)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            Hist_yequ = cv2.calcHist(yuv, [0], None, [256], [0, 256])
            histGraph_yequ = HistIP.ShowGrayHist(Hist_yequ, [255, 255, 255])
            HistIP.ImShow("y_equ", histGraph_yequ)
            EqualizeColor = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return EqualizeColor

    # @staticmethod
    # def HistMatching(SrcImg, RefImg, CType=ColorType.USE_HSV):
