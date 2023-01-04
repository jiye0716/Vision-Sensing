import cv2IP
import numpy as np
import random
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
ip = cv2IP.ConIP
img = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/tree.jpg")
# 平滑
# blur = ip.Smooth2D(img, 7, SmType=1)
# box = ip.Smooth2D(img, 7, SmType=2)
# gauss = ip.Smooth2D(img, 7, SmType=3)
# median = ip.Smooth2D(img, 7, SmType=4)
# bilateral = ip.Smooth2D(img, 7, SmType=5)
# ip.ImShow("INPUT", img)
# ip.ImShow("BLUR", blur)
# ip.ImShow("box", box)
# ip.ImShow("gauss", gauss)
# ip.ImShow("median", median)
# ip.ImShow("bilateral", bilateral)
# # cv2.waitKey(0)

# # 邊緣偵測
# absx,absy,dst = ip.EdgeDetect(img, 1)
# canny = ip.EdgeDetect(img, 2)
# scharr = ip.EdgeDetect(img, 3)
# laplace = ip.EdgeDetect(img, 4)
color_sobel = ip.EdgeDetect(img, 5)
# ip.ImShow("INPUT", img)
# ip.ImShow('sobel_absx', absx)
# ip.ImShow('sobel_absy', absy)
# ip.ImShow('sobel_dst', dst)
# ip.ImShow("canny", canny)
# ip.ImShow("scharr", scharr)
# ip.ImShow("laplace", laplace)
ip.ImShow("color_sobel", color_sobel)
# # cv2.waitKey(0)

# # 二維卷積
# Robert = ip.RobertOperator(img)
# Prewitt = ip.PrewittOperator(img)
# KirschMask = ip.KirschMaskOperator(img)
# cv.imshow('Input', img)
# cv.imshow('Robert', Robert)
# cv.imshow('Prewitt', Prewitt)
# cv.imshow('KirschMask', KirschMask)
# # cv2.waitKey(0)

# # 銳化
# LAPLACE_TYPE1 = ip.ImSharpening(img, 1)
# LAPLACE_TYPE2 = ip.ImSharpening(img, 2)
# LOG = ip.ImSharpening(img, 3)
# USM = ip.ImSharpening(img, 4)
# cv.imshow('img', img)
# cv.imshow('LAPLACE_TYPE1', LAPLACE_TYPE1)
# cv.imshow('LAPLACE_TYPE2', LAPLACE_TYPE2)
# cv.imshow('LOG', LOG)
# cv.imshow('USM', USM)
cv2.waitKey(0)
