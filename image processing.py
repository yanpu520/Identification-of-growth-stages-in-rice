#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


src = cv2.imread('shui.jpg')
cv2.namedWindow('src', 0)
cv2.resizeWindow('src', 900, 1000)
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()


# In[5]:


# 转换为浮点数进行计算
fsrc = np.array(src, dtype=np.float32) / 255.0
(b,g,r) = cv2.split(fsrc)
gray = 2 * g - b - r


# In[6]:


# 求取最大值和最小值
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)


# In[7]:


# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
plt.plot(hist)
plt.show()
cv2.waitKey()


# In[17]:


# 转换为u8类型，进行otsu二值化
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, 95, 255,cv2.THRESH_BINARY)
# plt.savefig("C:/Users/Admin/Desktop/1.jpg")
cv2.imwrite('1.png',bin_img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
cv2.namedWindow('bin_img', 0)
cv2.resizeWindow('bin_img', 700, 700)
cv2.imshow('bin_img', bin_img)
cv2.waitKey()
cv2.destroyAllWindows()


# In[86]:


#(3, 3)表示高斯滤波器的长和宽都为3，1.3表示滤波器的标准差
out=cv2.GaussianBlur(bin_img,(3,3),1.3)
cv2.namedWindow('result', 0)
cv2.resizeWindow('result', 700, 700)
cv2.imwrite('out.jpg',out)
cv2.imshow('result',out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[77]:


src=cv2.imread('1.png',cv2.IMREAD_UNCHANGED)


# In[78]:


kernel=np.ones((5,5),np.uint8)


# In[79]:


erosion=cv2.erode(src,kernel)
result = cv2.dilate(erosion, kernel)


# In[80]:


cv2.namedWindow('result', 0)
cv2.resizeWindow('result', 700, 700)
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


img = cv2.imread('bbe.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[5]:


img = cv2.imread('bbe.png')
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]


# In[7]:


cv2.namedWindow('b', 0)
cv2.resizeWindow('b', 700, 1000)
cv2.imwrite('bgray.jpg',b)
cv2.imshow("b", b)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


cv2.namedWindow('g', 0)
cv2.resizeWindow('g', 700, 1000)
cv2.imwrite('ggray.jpg',g)
cv2.imshow("g", g)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


cv2.namedWindow('r', 0)
cv2.resizeWindow('r', 700, 1000)
cv2.imwrite('rgray.jpg',r)
cv2.imshow("r", r)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[19]:


src = cv2.imread('rgray.jpg')
plt.hist(src.ravel(), 256)
plt.show()


# In[20]:


src = cv2.imread('ggray.jpg')
plt.hist(src.ravel(), 256)
plt.show()


# In[21]:


src = cv2.imread('bgray.jpg')
plt.hist(src.ravel(), 256)
plt.show()


# In[ ]:




