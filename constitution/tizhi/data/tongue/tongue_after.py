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
from unet.unet_model import UNet


def preprocess3(img):
    img = np.asarray(img)[:, :, :3]
   
    max_values = np.max(img, axis=(0, 1))
    min_values = np.min(img, axis=(0, 1))
    img = (img - min_values) / (max_values - min_values)
    mean_values = np.mean(img, axis=(0, 1)) 
    std_values = np.std(img, axis=(0, 1))
    img = (img - mean_values) / std_values
    img = img.transpose((2, 0, 1)).astype(np.float32)
    return img


best_epoch=100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2, bilinear=False)
model.to(device=device)
# state_dict = torch.load(f"/home/wangyifan/Practice/tcms/top_unet/unet_model_weight/unet3_weight_{best_epoch}.pth", map_location=device)  # 改动
state_dict = torch.load("/home/zhangxiaohan/constitution/tizhi/data/tongue/unet_model_weight/weights_epoch100.pth", map_location=device)
maskvalues = state_dict.pop("mask_values", [0, 1])
model.load_state_dict(state_dict)
model.eval()

vit = ViTModel.from_pretrained('/home/zhangxiaohan/constitution/Tools/vit_base_patch16_224_in21k')

def process_image(path):

    image = Image.open(path)
    image = image.resize((683, 512), resample=Image.BICUBIC)

    img = torch.from_numpy(preprocess3(image)) 
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(img).cpu() 
        mask = output.argmax(dim=1) 
    mask = mask[0].long().squeeze().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (224, 224)) 
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) 

    image = image.resize((224, 224), resample=Image.BICUBIC)
    image = np.asarray(image)[:, :, :3]
    image = image * mask  

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image = np.transpose(((image / 255.0 - mean) / std).astype(np.float32), (2, 0, 1))
    image = image.reshape([1] + list(image.shape)).astype(np.float32) 
    with torch.set_grad_enabled(False):
        feats = vit(torch.tensor(image)).pooler_output.squeeze(0)
        return feats


if __name__ == "__main__":

    # df = pd.read_csv("/home/sharing/disk1/liuhuilin/constitution/Constitution_English.csv")
    df = pd.read_csv("/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv")

    feats = {}

    for i in trange(df.shape[0]):
        id = int(df.iloc[i]['tongue_number'])
        img_name = id

        # top_path = f'/home/sharing/disk1/liuhuilin/constitution/tongue/{img_name}.jpg'
        top_path = f'/home/zhangxiaohan/constitution/data/tongue/{img_name}.jpg'

        if os.path.exists(top_path):
            feats[id] = process_image(top_path)
    
    with open(f"/home/zhangxiaohan/constitution/tizhi/data/tongue/feature/feature_after/tongue_after_zxh.pkl", 'wb') as f:
        pickle.dump(feats, f) 