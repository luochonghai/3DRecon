# -*- coding:utf-8 -*-
import cv2
cap = cv2.VideoCapture("test.mp4")# 获取一个视频打开cap 1 file name
isOpened = cap.isOpened# 判断是否打开‘
print(isOpened)
fps = cap.get(cv2.CAP_PROP_FPS)#帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#w h
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps,width,height)
i = 0
while(isOpened):
    if i == 900:#这里之后应该调整为帧率向上取整*视频时长
        break
    else:
        i = i+1
    (flag,frame) = cap.read()# 读取每一张 flag frame 
    fileName = './mp4tojpg/image'+str(i)+'.jpg'
    print(fileName)
    if flag == True:
        cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])
print('end!')
