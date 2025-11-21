import argparse
import imp
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from vit_model import vit_base_patch16_224
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='/home/zhangxiaohan/constitution/data/face',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='min',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard/0.1/0.3/0.5/0.7/0.9')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap_ori = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap_ori) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return heatmap_ori,np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()
    #vit
    model = vit_base_patch16_224(num_classes=2)
    weights_path = "/home/zhangxiaohan/constitution/explain/best_f1_weight.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # save_path_heatmap = os.path.join('/home/zhangxiaohan/constitution/explain/result/',"heatmap_"+args.head_fusion+"_"+str(args.discard_ratio))
    save_path_visualization = os.path.join('/home/zhangxiaohan/constitution/explain/result/',"visualization_"+args.head_fusion+"_"+str(args.discard_ratio))
    # if not os.path.exists(save_path_heatmap):
    #     os.makedirs(save_path_heatmap)
    if not os.path.exists(save_path_visualization):
        os.makedirs(save_path_visualization)

    for filename in os.listdir(args.image_path):
        print(filename)
        img_name = os.path.join(args.image_path,filename)

        img = Image.open(img_name)
        img = img.resize((224, 224))
        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = "attention_rollout_{:.3f}_{}.jpg".format(args.discard_ratio, args.head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, argjpgs.category_index)
            name = "grad_rollout_{}_{:.3f}_{}.jpg".format(args.category_index,
                args.discard_ratio, args.head_fusion)

        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        heatmap_ori,visualization = show_mask_on_image(np_img, mask)

        # cv2.imwrite("/home/liuhuilin/try/input.png", np_img)
        # cv2.imwrite(os.path.join(save_path_heatmap,filename),heatmap_ori)
        cv2.imwrite(os.path.join(save_path_visualization,filename),visualization)
        # cv2.imwrite(os.path.join(save_path_visualization,filename),visualization)
        # cv2.waitKey(-1)