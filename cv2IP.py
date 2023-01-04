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


class SmoothType(enum.IntEnum):
    BLUR = 1
    BOX = 2
    GAUSSIAN = 3
    MEDIAN = 4
    BILATERAL = 5


class EdgeType(enum.IntEnum):
    SOBEL = 1
    CANNY = 2
    SCHARR = 3
    LAPLACE = 4
    COLOR_SOBEL = 5


class SharpType(enum.IntEnum):
    LAPLACE_TYPE1 = 1
    LAPLACE_TYPE2 = 2
    SECOND_ORDER_LOG = 3
    UNSHARP_MASK = 4


class ConIP(BaseIP):
    # 影像平滑
    @staticmethod
    def Smooth2D(SrcImg, ksize, SmType=SmoothType.BLUR):
        source = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2RGB)
        if(SmType == SmoothType.BLUR):
            output = cv2.blur(source, (ksize, ksize))  # kernel 大小為 5x5
        if(SmType == SmoothType.BOX):
            # 第二個引數的-1表示輸出影象使用的深度與輸入影象相同
            output = cv2.boxFilter(source, -1, (ksize, ksize))
        if(SmType == SmoothType.GAUSSIAN):
            output = cv2.GaussianBlur(source, (ksize, ksize), 0.0)
        if(SmType == SmoothType.MEDIAN):
            output = cv2.medianBlur(source, ksize)
        if(SmType == SmoothType.BILATERAL):
            output = cv2.bilateralFilter(
                source, d=0, sigmaColor=100, sigmaSpace=10)
            # d：鄰域直徑
            # sigmaColor：顏色標準差 ，引數值較大時意味著在畫素點領域內的更多的顏色會被混合在一起
            # sigmaSpace：空間標準差，引數的較大值意味著更遠的畫素將與相互影響，只要它們的顏色足夠相近
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
    # 邊緣偵測

    @staticmethod
    def EdgeDetect(SrcImg, EdType=EdgeType.SOBEL):
        if(EdType == EdgeType.SOBEL):
            Source = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(Source, cv2.CV_16S, 1, 0)  # 垂直偵測
            sobel_y = cv2.Sobel(Source, cv2.CV_16S, 0, 1)  # 水平偵測
            absx = np.abs(sobel_x).astype('uint8')
            absy = np.abs(sobel_y).astype('uint8')
            dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
            return absx, absy, dst
        if(EdType == EdgeType.CANNY):
            kernel = 3
            min_threshold = 50  # 最小門檻值
            max_threshold = 150  # 最大門檻值
            Source = cv2.GaussianBlur(SrcImg, (kernel, kernel), 0)
            output = cv2.Canny(Source, min_threshold, max_threshold)
            return output
        if(EdType == EdgeType.SCHARR):
            Source = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
            x = cv2.Scharr(Source, cv2.CV_16S, 1, 0)
            y = cv2.Scharr(Source, cv2.CV_16S, 0, 1)
            absx = np.abs(x).astype('uint8')
            absy = np.abs(y).astype('uint8')
            output = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
            return output
        if(EdType == EdgeType.LAPLACE):
            Source = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
            laplacian_img = cv2.Laplacian(Source, cv2.CV_16S, ksize=3)
            output = np.abs(laplacian_img).astype('uint8')
            return output
        if(EdType == EdgeType.COLOR_SOBEL):
            b, g, r = cv2.split(SrcImg)
            bx = cv2.Sobel(b, cv2.CV_16S, 1, 0)
            by = cv2.Sobel(b, cv2.CV_16S, 0, 1)
            gx = cv2.Sobel(g, cv2.CV_16S, 1, 0)
            gy = cv2.Sobel(g, cv2.CV_16S, 0, 1)
            rx = cv2.Sobel(r, cv2.CV_16S, 1, 0)
            ry = cv2.Sobel(r, cv2.CV_16S, 0, 1)
            absbx = np.abs(bx).astype('uint8')
            absby = np.abs(by).astype('uint8')
            absgx = np.abs(gx).astype('uint8')
            absgy = np.abs(gy).astype('uint8')
            absrx = np.abs(rx).astype('uint8')
            absry = np.abs(ry).astype('uint8')
            outputb = cv2.addWeighted(absbx, 0.5, absby, 0.5, 0)
            outputg = cv2.addWeighted(absgx, 0.5, absgy, 0.5, 0)
            outputr = cv2.addWeighted(absrx, 0.5, absry, 0.5, 0)
            output = cv2.merge([outputb, outputg, outputr])
            return output

    # 二維卷積
    @staticmethod
    def RobertOperator(SrcImg):
        SrcImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
        # Roberts 算子
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        x = cv2.filter2D(SrcImg, cv2.CV_16S, kernelx)
        y = cv2.filter2D(SrcImg, cv2.CV_16S, kernely)
        absx = cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        output = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        return output

    @staticmethod
    def PrewittOperator(SrcImg):
        SrcImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
        # Prewitt 算子
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        # 轉 uint8 ,圖像融合
        x = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, kernelx))
        y = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, kernely))
        output = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        return output

    @staticmethod
    def KirschMaskOperator(SrcImg):
        SrcImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
        G__N = np.array(([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]))
        G_NW = np.array(([[5, -3, -3], [5, 0, -3], [5, -3, -3]]))
        G__W = np.array(([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]))
        G_SW = np.array(([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]))
        G__S = np.array(([[5, 5, 5], [-3, 0, -3], [-3, -3, -3, ]]))
        G_SE = np.array(([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]))
        G__E = np.array(([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]))
        G_NE = np.array(([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]))

        G__N_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G__N))
        G_NW_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G_NW))
        G__W_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G__W))
        G_SW_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G_SW))
        G__S_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G__S))
        G_SE_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G_SE))
        G__E_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G__E))
        G_NE_conv = cv2.convertScaleAbs(cv2.filter2D(SrcImg, cv2.CV_16S, G_NE))

        output = cv2.max(G__N_conv, cv2.max(G_NW_conv, cv2.max(G__W_conv, cv2.max(
            G_SW_conv, cv2.max(G__S_conv, cv2.max(G_SE_conv, cv2.max(G__E_conv, G_NE_conv)))))))
        return output

    @staticmethod
    def Conv2D(SrcImg, kernel):
        out = cv2.filter2D(SrcImg, -1, kernel)
        return out
    # 影像銳利化

    @staticmethod
    # , SmoothType=SmoothType.BILATERAL
    def ImSharpening(SrcImg, SpType=SharpType.UNSHARP_MASK):
        if(SpType == SharpType.LAPLACE_TYPE1):
            kernel = np.array(
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
            output = ConIP.Conv2D(SrcImg, kernel)
            return output
        if(SpType == SharpType.LAPLACE_TYPE2):
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            output = ConIP.Conv2D(SrcImg, kernel)
            return output
        if(SpType == SharpType.SECOND_ORDER_LOG):
            img = cv2.GaussianBlur(SrcImg, (3, 3), 0.0)
            laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
            output = cv2.convertScaleAbs(laplacian)
            return output
        if(SpType == SharpType.UNSHARP_MASK):
            img = cv2.GaussianBlur(SrcImg, (0, 0), 5)
            sharpened = float(1.0 + 1) * SrcImg - float(1.0) * img
            sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
            sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
            output = sharpened.round().astype(np.uint8)
            return output
