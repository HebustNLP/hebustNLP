import pandas as pd
from pathlib import Path
import os, sys
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold


class TCMDataset(Dataset):
    def __init__(self, root_path="./data",csv_path="", label_name="", modality=[], mode="train", seed=2023, k=5, test=0):
        super().__init__()
        self.root_path = root_path
        self.csv_path = csv_path
        self.label_name = label_name
        self.modality = modality
        self.data = pd.read_csv(csv_path, encoding="utf-8")
       
        if 'face' in modality:
            # /home/wangyifan/SZY/face/new_feature/face0617.pkl  /home/liuhuilin/multimodal/constitution/feature/face_mediapipe.pkl
            # /home/wangyifan/SZY/face/swin_feature/swin_face.pkl
            with open('/home/zhangxiaohan/constitution/tizhi/fusion/data/feature/face_mediapipe_zxh.pkl', 'rb') as f:
                self.face = pickle.load(f)
            drop_list = []
            for i, v in self.data['face_number'].items():
                if v not in self.face.keys():
                    drop_list.append(i)
            self.data = self.data.drop(drop_list)
        if 'tongue' in modality:
            # /home/wangyifan/SZY/tongue/new_feature/tongue0617.pkl  /home/liuhuilin/multimodal/constitution/feature/tongue_unet_improve2.pkl
            # /home/wangyifan/SZY/tongue/swin_feature/swin_tongue.pkl
            with open('/home/zhangxiaohan/constitution/tizhi/fusion/data/feature/tongue_mediapipe_zxh.pkl', 'rb') as f:
                self.tongue = pickle.load(f) 
            drop_list = []
            for i, v in self.data['tongue_number'].items(): 
                if v not in self.tongue.keys(): 
                    drop_list.append(i)  
            self.data = self.data.drop(drop_list) 
        

        self.data = self.data.reset_index(drop=True)


        cv_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) 
        train_idx, val_idx, test_idx = [], [], []
        for i, (_, fold_index) in enumerate(cv_split.split(self.data.index, self.data[self.label_name])):
            if i == test:
                test_idx.extend(fold_index)
            else:
                train_idx.extend(fold_index)
        train_idx, val_idx, _, _ = train_test_split(self.data.loc[train_idx].index, self.data.loc[train_idx][self.label_name], test_size=0.25, random_state=seed, stratify=self.data.loc[train_idx][self.label_name])
        if mode == 'train': 
            self.data = self.data.loc[train_idx].reset_index(drop=True)
        elif mode == 'val':
            self.data = self.data.loc[val_idx].reset_index(drop=True)
        else:
            self.data = self.data.loc[test_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {
            'id': self.data['ID'][index],
            'ask': torch.tensor(self.data.drop(columns=["ID", "face_number", "tongue_number", "constitution"]).iloc[index].values, dtype=torch.float32),

            'label_name': torch.tensor(self.data[self.label_name][index], dtype=torch.long)
        }
        if 'face' in self.modality:
            sample['face'] = self.face[self.data['face_number'][index]]
        if 'tongue' in self.modality:
            sample['tongue'] = self.tongue[self.data['tongue_number'][index]]
        
        return sample


def TCMDataloader(root_path="./data", csv_path="", label_name="", modality=[], seed=2023, batch_size=512, num_workers=0, k=5, test=0):
    train_set = TCMDataset(root_path, csv_path,label_name, modality, mode="train", k=k, test=test, seed=seed)
    val_set = TCMDataset(root_path,  csv_path,label_name, modality, mode="val", k=k, test=test, seed=seed)
    test_set = TCMDataset(root_path, csv_path, label_name, modality, mode="test", k=k, test=test, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
