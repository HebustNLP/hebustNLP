import numpy as np
from PIL import Image,ImageDraw
import imgviz
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from unet.unet_model import UNet
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 全局归一化
def preprocess1(img):
    img = np.asarray(img)[:, :, :3]  # 获取每个像素的前三个通道 因为有可能存在第四个通道
    img = img / 255.0  # 进行归一化操作 像素值都介于(0,255)之间
    img = img.transpose((2, 0, 1)).astype(np.float32)  # 对通道数顺序进行调整 把通道数放到前面
    return img

# 全局归一化+全局标准化
def preprocess2(img):
    img = np.asarray(img)[:, :, :3]
    img = img / 255.0
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    img = (img - mean) / std
    img = img.transpose((2, 0, 1)).astype(np.float32)  # 将通道数放到前面
    return img

# 局部归一化+局部标准化
def preprocess3(img):
    img = np.asarray(img)[:, :, :3]
    # 在图像的宽度和高度方向上计算每个通道在整个图像中的最大像素值  对于红色通道,在整张特征图中找出所有像素的红色值中的最大值(蓝色绿色同理)
    max_values = np.max(img, axis=(0, 1))  # [184 128 126]
    min_values = np.min(img, axis=(0, 1))  # [47 53 38]
    img = (img - min_values) / (max_values - min_values)
    mean_values = np.mean(img, axis=(0, 1))  # 计算每个通道的均差 [0.70377443 0.60042903 0.52263415]
    std_values = np.std(img, axis=(0, 1))  # 计算每个通道的标准差 [0.17425166 0.16350889 0.19639324]
    img = (img - mean_values) / std_values
    img = img.transpose((2, 0, 1)).astype(np.float32)
    return img

best_epoch=100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 只对图片进行归一化训练出来的网络
net1 = UNet(n_channels=3, n_classes=2, bilinear=False)  # 创建一个模型实例 通道数为3 分类为2分类 是否采用双线性插值
net1.to(device=device)
# state_dict = torch.load(f"/home/wangyifan/Practice/tcms/top_unet/unet_model_weight/unet1_weight_{best_epoch}.pth", map_location=device)
state_dict = torch.load(f"/home/zhangxiaohan/constitution/tizhi/data/tongue/unet_model_weight/weights_epoch{best_epoch}.pth", map_location=device)
maskvalues = state_dict.pop("mask_values", [0, 1])  # 从模型权重中取出为mask_values的值,如果不存在则返回[0,1]  为了获取特定的模型配置或参数
net1.load_state_dict(state_dict)  # 将加载的模型权重加载到net1模型上
net1.eval()  # 设置为评估模式

# 对图片进行归一化+全局标准化训练出来的网络
net2 = UNet(n_channels=3, n_classes=2, bilinear=False)
net2.to(device=device)
# state_dict = torch.load(f"/home/wangyifan/Practice/tcms/top_unet/unet_model_weight/unet2_weight_{best_epoch}.pth", map_location=device)
state_dict = torch.load(f"/home/zhangxiaohan/constitution/tizhi/data/tongue/unet_model_weight/weights_epoch{best_epoch}.pth", map_location=device)
maskvalues = state_dict.pop("mask_values", [0, 1])
net2.load_state_dict(state_dict)
net2.eval()

# 对图片进行局部归一化+局部标准化训练出来的网络
net3 = UNet(n_channels=3, n_classes=2, bilinear=False)
net3.to(device=device)
# state_dict = torch.load(f"/home/wangyifan/Practice/tcms/top_unet/unet_model_weight/unet3_weight_{best_epoch}.pth", map_location=device)
state_dict = torch.load(f"/home/zhangxiaohan/constitution/tizhi/data/tongue/unet_model_weight/weights_epoch100.pth", map_location=device)
maskvalues = state_dict.pop("mask_values", [0, 1])
net3.load_state_dict(state_dict)
net3.eval()

path = "/home/zhangxiaohan/constitution/data/tongue/201509070004.jpg"
# path = "/home/sharing/disk1/lisongze/cm/upload/tongue/201509070004.jpg"
# path = "/home/sharing/disk1/lisongze/cm/upload/tongue/201509070055.jpg"
# path = '/home/sharing/disk1/lisongze/cm/upload/tongue/201509110042.jpg'
image = Image.open(path)

# # image.save('origin.jpg')
# w, h = image.size  # w=900 h=1200  print(image.mode) # GRB三通道
# left = 0
# right = w
# top = 0
# bottom = h // 1.8  # 整除
# image = image.crop((left, top, right, bottom))  # 保留了图像上半部分的666像素
# # print(np.asarray(image).shape)  #  (666, 900, 3)
# # image.save('image.jpg')  # 保存图像

image = image.resize((683, 512), resample=Image.BICUBIC)

# 用net1分割
img1 = torch.from_numpy(preprocess1(image))  # 对图像进行预处理后转化为pytorch的Tensor张量格式
# print(img1.shape)  # [3,666,900]
img1 = img1.unsqueeze(0)  # 对图像维度进行扩展 在最前面增加一个维度 从而符合模型输入要求
# print(img1.shape)  # [1,3,666,900]
img1 = img1.to(device=device, dtype=torch.float32)  # 移动到GPU上进行计算
with torch.no_grad():
    output = net1(img1).cpu()  # 将图像输入到net1中得到output后 再移动到cpu上进行后续处理
    # print(output)  # 四维张量  [1, 2, 666, 900]  2表示通道数 二分类任务

    # output = F.interpolate(output, (512, 512), mode='bilinear')  # 利用双线性插值调整输出结果大小
    
    # print(output.shape)  # [1,2,512,512]
    if net1.n_classes > 1:  # 上面定义的net1.n_classes=2
        mask1 = output.argmax(dim=1)  # 获取在通道维度上最大值所对应的类别标签  每个类别对应一个通道,每个通道的值表示模型认为该样本属于该类别的概率
        # print(mask1.shape)  # mask1是一个三维张量 [1,512,512] 不全是零 只是终端显示出来的都是零而已
        # print(torch.sum(mask1).item())  # 109859
    else:
        mask1 = torch.sigmoid(output) > 0.5  # 根据阈值0.5进行二值化处理,得到一个二值化的预测结果

mask1 = mask1[0].long().squeeze().numpy() # mask1[0]对于[1,512,512] 如果是[3,512,512]就是为了取出第一个[512,512]  squeeze()将维度大小为1的维度去除
# print(mask1)  # 二维张量

# 用net2分割
img2 = torch.from_numpy(preprocess2(image))  # 对图像进行预处理 归一化+全局标准化
img2 = img2.unsqueeze(0)  # 维度扩展
img2 = img2.to(device=device, dtype=torch.float32)
with torch.no_grad():
    output = net2(img2).cpu()
    # print(output.shape)  # 四维张量 [1,2,666,900]

    # output = F.interpolate(output, (512, 512), mode='bilinear')
    
    # print(output.shape)  #  [1, 2, 512, 512]
    if net2.n_classes > 1:  #  net2.n_classes=2
        mask2 = output.argmax(dim=1)
        # print(mask2.shape)  # mask1是一个全零的三维张量[1,512,512]
    else:
        mask2 = torch.sigmoid(output) > 0.5
mask2 = mask2[0].long().squeeze().numpy()
# print(mask2.shape)  # (512,512)

# 用net3分割
img3 = torch.from_numpy(preprocess3(image))  # 对图像进行局部归一化+局部标准化
img3 = img3.unsqueeze(0)  # 维度扩展
img3 = img3.to(device=device, dtype=torch.float32)
with torch.no_grad():
    output = net3(img3).cpu()

    # output = F.interpolate(output, (512, 512), mode='bilinear')
    
    # print(output.shape)
    if net3.n_classes > 1:
        mask3 = output.argmax(dim=1)
    else:
        mask3 = torch.sigmoid(output) > 0.5
mask3 = mask3[0].long().squeeze().numpy()

# image = image.resize((512, 512))
img = np.asarray(image)  # 将图像转换为numpy数组

# 将原图像转换为灰色图像 但是将标签数据mask1转换为彩色图像方便展示
seg_img1 = imgviz.label2rgb(label=mask1, image=imgviz.asgray(img))  # label2rgb函数要根据mask1中的类别标签信息生成不同的颜色来分割图像
seg_img2 = imgviz.label2rgb(label=mask2, image=imgviz.asgray(img))
seg_img3 = imgviz.label2rgb(label=mask3, image=imgviz.asgray(img))

# plt.imsave('seg_img1.png', seg_img1)
# plt.imsave('seg_img2.png', seg_img2)
plt.imsave('/home/zhangxiaohan/constitution/tizhi/data/tongue/image_example/seg_img3_zxh.jpg', seg_img3)