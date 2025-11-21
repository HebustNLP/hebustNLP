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

def process_image(path):
    image=cv2.imread(path)
    image=cv2.resize(image,(224,224))

    image=image[:,:,::-1]  # 将图像进行通道顺序转换(从BGR到RGB)
    
    # 对图像进行标准化处理 有助于图像更好的学习特征
    mean=[0.5,0.5,0.5]
    std=[0.5,0.5,0.5]
    image = np.transpose(((image / 255.0 - mean) / std), (2, 0, 1))  # 对图像进行归一化操作  将原始数组的维度顺序从(0,1,2)变换为(2,0,1) (h,w,c)->(c,h,w) 符合模型的输入
    image = image.reshape([1] + list(image.shape)).astype(np.float32)  # 将图像的通道维度放到最前面,并添加一个额外的维度,使其成为形状为[1,C,H,W]的张量
    with torch.set_grad_enabled(False):  # 关闭梯度计算
        feats = vit(torch.tensor(image)).pooler_output.squeeze(0)  # 使用预训练的ViT模型对图像进行特征提取
        return feats  # 返回提取得到的特征feats


if __name__=="__main__":
    
    df=pd.read_csv('/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv')
    face={}

    for i in trange(df.shape[0]):  # df.shape输出为(1084, 47)  故df.shape[0]输出为1084  trange循环的时候显示进度条
        id=df.iloc[i]['face_number']  # 每次循环获取第i行 列名为face_number的数据
        img_name=id

        face_path=f'/home/zhangxiaohan/constitution/data/face/{img_name}.jpg'

        if os.path.exists(face_path):  # 如果指定的face_path路径存在文件
            face[id]=process_image(face_path) # 在process_image中处理图像 存到字典中以id作为键
        
    # 将提取出的人脸特征数据face_feats保存到指定路径下的 pickle 文件
    with open(f'/home/zhangxiaohan/constitution/tizhi/data/face/feature/feature_before/face_before_zxh.pkl','wb') as f:  # 以二进制写入
        pickle.dump(face,f)