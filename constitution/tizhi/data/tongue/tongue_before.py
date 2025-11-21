import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os, sys
import pandas as pd
import pickle
from tqdm import trange
import cv2
from transformers import ViTModel

# os.chdir(sys.path[0])  # 确保当前工作目录就是Python脚本所在的目录

vit = ViTModel.from_pretrained('/home/zhangxiaohan/constitution/Tools/vit_base_patch16_224_in21k')

def process_image(path):
    # 裁剪图像
    image = Image.open(path)
    w, h = image.size
    left = 0
    right = w
    top = 0
    bottom = h // 1.8
    image = image.crop((left, top, right,  bottom))

    image = image.resize((224, 224), resample=Image.BICUBIC)  # 将图像通过双三次插值调整为224x224像素
    image = np.asarray(image)[:, :, :3]  # [224,224,3] 将image转换为numpy数组且仅保留前三个通道

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image = np.transpose(((image / 255.0 - mean) / std).astype(np.float32), (2, 0, 1))  # # 对图像进行归一化操作
    image = image.reshape([1] + list(image.shape)).astype(np.float32)  # 将图像的通道维度放到最前面,并添加一个额外的维度,使其成为形状为[1,C,H,W]的张量
    
    with torch.set_grad_enabled(False): # 不进行梯度计算
        feats = vit(torch.tensor(image)).pooler_output.squeeze(0)  # 使用vit模型提取图像特征,并返回特征向量  squeeze(0)除去第一维
        return feats

if __name__ == "__main__":
    df = pd.read_csv("/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv")
    feats = {}

    for i in trange(df.shape[0]):
        id = df.iloc[i]['tongue_number']
        img_name = id

        top_path = f'/home/zhangxiaohan/constitution/data/tongue/{img_name}.jpg'

        if os.path.exists(top_path):
            feats[id] = process_image(top_path)
    
    # 将提取出的舌上特征数据feats保存到指定路径下 保存为pickle文件
    with open(f"/home/zhangxiaohan/constitution/tizhi/data/tongue/feature/feature_before/tongue_before_zxh.pkl", 'wb') as f:
        pickle.dump(feats, f)