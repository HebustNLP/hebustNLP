import cv2
import os
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import trange
import pickle
from sklearn.decomposition import PCA

# 初始化 FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,      # 静态图像模式
    max_num_faces=1,             # 只检测一张脸
    refine_landmarks=True,       # 是否提取精细 landmarks
    min_detection_confidence=0.5
)

def extract_mediapipe_features(image_path):
    """用 MediaPipe 提取面部 468 个关键点 (x, y, z) 作为特征"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] 读取失败: {image_path}")
        return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print(f"[WARN] 未检测到人脸: {image_path}")
        return None

    # 取第一张脸的468个关键点
    landmarks = results.multi_face_landmarks[0].landmark
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])  # 每个关键点 3 个值

    return np.array(features, dtype=np.float32)  # shape=(1404,)

if __name__ == "__main__":
    df = pd.read_csv("/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv")
    raw_feats = []
    ids = []

    for i in trange(df.shape[0]):
        img_id = int(df.iloc[i]['face_number'])
        face_path = f'/home/zhangxiaohan/constitution/data/face/{img_id}.jpg'

        if os.path.exists(face_path):
            feats = extract_mediapipe_features(face_path)
            if feats is not None:
                raw_feats.append(feats)
                ids.append(img_id)

    raw_feats = np.array(raw_feats)  # shape=(N, 1404)

    # 用 PCA 压缩到 768 维
    print("[INFO] 正在用 PCA 将特征从 1404 维降到 768 维...")
    pca = PCA(n_components=768, random_state=42)
    reduced_feats = pca.fit_transform(raw_feats)

    # 转成字典形式保存
    face_feats = {ids[i]: reduced_feats[i] for i in range(len(ids))}

    save_path = '/home/zhangxiaohan/constitution/tizhi/data/face/feature/feature_mediapipe/face_mediapipe_zxh.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(face_feats, f)

    print(f"[INFO] 提取完成，共保存 {len(face_feats)} 条特征 -> {save_path}")
