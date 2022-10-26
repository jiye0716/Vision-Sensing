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
back = ip.ImRead(r"C:/testAI/Vision-Sensing/Imgs/bridge.jpg")
print(back.shape)  # 圖片大小
# back = cv2.resize(back, (1000, 600))
# # ImDim = np.shape(fore_cow)
# # if (ImDim[0] != back.shape[0] or ImDim[1] != back.shape[1]):
# #     back = cv2.resize(back, (ImDim[1], ImDim[0]))

# k = 0


# back = np.float32(back)
# out = ip.DoBlending(fore_cow, back, alpha_cow)
# out = np.uint8(out)
ip.ImWindow("AlphaBlending Result")
ip.ImShow("AlphaBlending Result", back)
