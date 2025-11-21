# import dlib
import cv2
import os, sys
import numpy as np
import mediapipe as mp
import torch
import pandas as pd
from tqdm import trange
import pickle
from transformers import ViTImageProcessor, ViTModel


# 加载的是整个vit模型
vit = ViTModel.from_pretrained('/home/zhangxiaohan/constitution/Tools/vit_base_patch16_224_in21k')

# 468个关键点
mp_face_mesh = mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh()


image=cv2.imread('/home/zhangxiaohan/constitution/data/face/201509070003.jpg')
image=cv2.resize(image,(224,224))  # (224,224,3)
rgb = image[:,:,::-1]  # (224,224,3)
result = face_mesh.process(rgb)  # 对RGB格式的图像进行人脸关键点检测,返回检测结果

# mask = np.zeros(image.shape[:2])  # 创建一个与输入图像大小相同的全零数组,用于存储生成的人脸掩码 (224,224)
mask=np.zeros_like(image)  # (224,224,3)
# cv2.imwrite('mask.jpg',mask)

points=[]
if result.multi_face_landmarks:  # 检查是否检测到了人脸
    for landmarks in result.multi_face_landmarks:  # 遍历检测到的多个关键点 每个landmarks对象代表一个人脸的关键点信息
        # print(landmarks) # 输出的每个landmark(对象)包含了三个值x,y,z  x表示关键点在人脸区域中的水平位置(横坐标) y表示纵坐标 z表示关键点的深度 即离相机的距离
        for landmark in landmarks.landmark: #  landmarks.landmark表示人脸关键点的具体坐标信息集合
            point = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))  # x坐标乘以图像宽度 y坐标乘以图像高度 将人脸关键点中的相对位置转化为具体的像素位置
            points.append(point)
            cv2.circle(image, (point[0], point[1]), 2, (0, 0, 255), -1)  
        hull = cv2.convexHull(np.array(points), returnPoints=True) # hull表示凸包(坐标)  通过输入点坐标 经过计算输出的是人脸信息的轮廓(凸包)的一系列坐标
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))  # mask为一张纯黑色图片 利用hull把hull坐标围成的区域变成白色 即人脸区域是白色的但是目前不显示人脸  其它区域是黑色的

# cv2.imwrite('mask1.jpg',mask)
results = cv2.bitwise_and(image, mask)  # image与mask的对应像素都非零results才非零 有一个为零即黑色 则results为黑色

# cv2.imwrite('image.jpg',image)
cv2.imwrite("/home/zhangxiaohan/constitution/tizhi/data/face/image_example/output_image_zxh.jpg", results)