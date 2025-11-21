import cv2
import os, sys
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange
import pickle
import timm
from swin_transformers import SwinTransformer

# 初始化 SwinTransformer
swin = SwinTransformer(pretrained=True)
swin.eval()

# 用于把 swin 输出映射到 768 维
# 假设 swin 输出维度是 out_dim（可以在第一次 forward 时动态确定）
reduce_layer = None

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# segment
def mask_468(image):
    rgb = image[:, :, ::-1]
    result = face_mesh.process(rgb)
    mask = np.zeros(image.shape[:2])
    points = []
    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            for landmark in landmarks.landmark:
                point = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                points.append(point)
            hull = cv2.convexHull(np.array(points), returnPoints=True)
            cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(np.uint8)

def process_image(path):
    global reduce_layer
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))

    mask = mask_468(image)
    if mask.sum() == 0:
        print('bad face')

    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    image = image[:, :, ::-1]

    face = image * mask
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    face = np.transpose(((face / 255.0 - mean) / std), (2, 0, 1))
    face = face.reshape([1] + list(face.shape)).astype(np.float32)

    with torch.no_grad():
        feats = swin(torch.tensor(face))  # 原始 swin 输出
        feats = feats.squeeze(0)

        # 初始化映射层
        if reduce_layer is None:
            out_dim = feats.shape[-1]
            reduce_layer = nn.Linear(out_dim, 768)
            reduce_layer.eval()

        feats_768 = reduce_layer(feats)
        return feats_768.numpy()

if __name__ == "__main__":
    df = pd.read_csv("/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv")
    face_feats = {}

    for i in trange(df.shape[0]):
        img_id = int(df.iloc[i]['tongue_number'])
        face_path = f'/home/zhangxiaohan/constitution/data/tongue/{img_id}.jpg'

        if os.path.exists(face_path):
            face_feats[img_id] = process_image(face_path)

    save_path = '/home/zhangxiaohan/constitution/tizhi/data/tongue/feature/feature_swin/tongue_swin_zxh.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(face_feats, f)

    print(f"[INFO] 提取完成，共保存 {len(face_feats)} 条特征 -> {save_path}")
