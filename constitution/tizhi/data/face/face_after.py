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


vit = ViTModel.from_pretrained('/home/zhangxiaohan/constitution/Tools/vit_base_patch16_224_in21k')

mp_face_mesh = mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh() 

# segment
def mask_468(image):
    rgb = image[:,:,::-1] 
    result = face_mesh.process(rgb) 
    mask = np.zeros(image.shape[:2])
    points = []
    if result.multi_face_landmarks:  
        for landmarks in result.multi_face_landmarks: 
           
            for landmark in landmarks.landmark: 
                point = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) 
                points.append(point)
            hull = cv2.convexHull(np.array(points), returnPoints=True)  # 闭包 
            cv2.fillConvexPoly(mask, hull, 1)  # 得到闭包包围的区域值为1 其它区域为0

    return mask.astype(np.uint8) 

def process_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(224,224))

    mask = mask_468(image) 
    cv2.imwrite("mask_zxh.jpg",mask)

    if mask.sum()==0:
        print('bad face')

    mask = np.repeat(mask[:,:,np.newaxis],3,axis=2)
    cv2.imwrite("mask2_zxh.jpg",mask)
     
    image = image[:,:,::-1] 

    face = image * mask  # 获取segment后区域的原像素值 颜色变化来自于通道数的改变
    cv2.imwrite("segment_face_zxh.jpg",face)

    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    face = np.transpose(((face / 255.0 - mean) / std), (2, 0, 1))
    face = face.reshape([1] + list(face.shape)).astype(np.float32)  
    
    with torch.set_grad_enabled(False): 
        feats = vit(torch.tensor(face)).pooler_output.squeeze(0) 
        return feats 


if __name__=="__main__":

    df = pd.read_csv("/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv")
    
    face_feats = {}

    for i in trange(df.shape[0]): 
        id = int(df.iloc[i]['face_number']) 
        img_name = id

        face_path = f'/home/zhangxiaohan/constitution/data/face/{img_name}.jpg'

        if os.path.exists(face_path):
            face_feats[id] = process_image(face_path)
           
    with open(f'/home/zhangxiaohan/constitution/tizhi/data/face/feature/feature_after/face_after_zxh.pkl','wb') as f: 
        pickle.dump(face_feats,f)