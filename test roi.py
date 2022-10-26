from telnetlib import IP
import cv2
import numpy as np
import cv2IP

ip = cv2IP.AlphaBlend()
# 讀取圖片 load image
img1 = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/cowherd and weaver.png")
img1 = cv2.resize(img1, (375, 250))
cow = img1[100:240, 10:150]
girl = img1[10:150, 210:350]  # 140*140

cow_split = ip.SplitAlpha(cow)
girl_split = ip.SplitAlpha(girl)
fore_cow = np.float32(cow_split[0])
fore_girl = np.float32(girl_split[0])
# # Alpha Channel
# ip.ImWindow("Alpha Channel")
# ip.ImShow("Alpha Channel", cow)
alpha_cow = np.float32(cow_split[1])/255
alpha_girl = np.float32(girl_split[1])/255


# back = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/bridge.jpg")

# print(back.shape)  # 圖片大小
i = 10
for i in range(10, 110, 10):
    back = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/bridge.jpg")
    ROI_cow = back[130:270, i:i+140, :]
    ROI_girl = back[130:270, 360-i: 500-i, :]
    ROI_cow = np.float32(ROI_cow)
    ROI_girl = np.float32(ROI_girl)
    out_cow = ip.DoBlending(fore_cow, ROI_cow, alpha_cow)
    out_girl = ip.DoBlending(fore_girl, ROI_girl, alpha_girl)
    out_cow = np.uint8(out_cow)
    out_girl = np.uint8(out_girl)
    back[130:270, i:i+140, :] = out_cow
    back[130:270, 360-i: 500-i, :] = out_girl
    ip.ImWindow("AlphaBlending Result")
    ip.ImShow("AlphaBlending Result", back)
