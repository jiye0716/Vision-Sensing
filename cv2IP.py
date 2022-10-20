import cv2 as cv
import numpy as np
import enum


class BaseIP(object):
    # 影像讀取、儲存與顯示函數
    @staticmethod
    def ImRead(filename):
        return cv.imread(filename, cv.IMREAD_UNCHANGED)
    # 影像儲存

    @staticmethod
    def ImWrite(filename, img):
        return cv.imwrite(filename, img)
    # 顯示影像

    @staticmethod
    def ImShow(winname, img):
        cv.imshow(winname, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # 創建顯示視窗

    @staticmethod
    def ImWindow(winname):
        cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)


class AlphaBlend(BaseIP):
    @staticmethod
    def SplitAlpha(SrcImg):
        b, g, r, a = cv.split(SrcImg)
        fore = cv.merge([b, g, r])
        alpha = cv.merge([a, a, a])
        return fore, alpha

    @staticmethod
    def DoBlending(Foreground, Background, Alpha):
        fore = cv.multiply(Foreground, Alpha)
        back = cv.multiply(Background, 1.0-Alpha)
        out = cv.add(fore, back)
        return out
