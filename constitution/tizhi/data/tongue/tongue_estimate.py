import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
import os
import time
# from dataloader import TCMDataloader
import logging
import os, sys
import csv
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle

os.chdir(sys.path[0])  # 确保当前工作目录就是Python脚本所在的目录

os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 改动
logger = logging.getLogger()

# 设置随机种子,确保在使用随机数时能够复现结果,增强实验的可复现性
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 模型构造 线性层
class Model(nn.Module):
    def __init__(self, num_classes=3, dim=[768, 256, 128]):
        super().__init__()
        layers = []
        dim.append(num_classes)  # dim=[768,256,128,3]

        for i in range(1, len(dim)): # 768 256 relu 256 128 relu 128 3
            layers.append(nn.Linear(dim[i - 1], dim[i]))
            if i < len(dim) - 1:
                # layers.append(nn.BatchNorm1d(dim[i]))
                layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers) # 顺序的将多个神经网络层组合在一起
    
    def forward(self, x):
        out = self.fc(x)
        return out


"""
__init__     把数据集(self.data 操作的主要是问诊和标签)随机的均匀的划分为训练集 测试集 验证集
__len__      返回数据集的样本数量 当调用len(实例化对象名)时 会返回数据集的样本数量
__getitem__  用于实现索引操作  当调用 实例化对象名[i]时 会索引到我们指定的第i个样本
"""
# 初始化数据集    
class TCMDataset(Dataset):
    def __init__(self, root_path="./data", image_path="", name="", mode="", seed=2023, k=5, test=0):
        super().__init__()  # 可以确保在子类初始化的同时也执行了父类的初始化方法 也就是把父类中的__init__()也进行了初始化
            
        self.root_path = root_path
        self.name = name    # 标签名
 
        # 加载问诊数据  1084×47 #   /home/sharing/disk1/lisongze/cm/upload/Constitution0410.csv
        self.data = pd.read_csv(Path('/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv'), encoding="utf-8")
        
        # 打开提取到的特征文件(字典)
        with open(image_path, 'rb') as f:
            self.img = pickle.load(f)  # 从文件数据中加载数据  一个字典 每个键(索引值)都对应这对应的图片的保存成的张量
        
        # 如果特征文件中没有这个病人对应的图片特征 则把data中对应这个病人的行删去
        drop_list = []
        for i, v in self.data['tongue_number'].items():  # 遍历new_id这一列 i为索引 v为new_id每行的值
            if v not in self.img.keys():  # 如果v没有在特征文件字典的键里面
                drop_list.append(i)  # 添加到删除列表中
        self.data = self.data.drop(drop_list)  # 根据索引值删除对应行的数据  得到新的数据集self.data
        self.data = self.data.reset_index(drop=True)  # 重置数据集索引   self.data为 1084×47

        # k折交叉验证
        cv_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)  # 创建一个StratifiedKFold对象
        train_idx, val_idx, test_idx = [], [], []  # 为了得到训练集、验证集和测试集的索引
        """
        cv_split()  表示分层k折交叉验证,与普通的k折交叉验证不同的是,它确保每个折叠中的类别比例与整个数据集中的类别比例保持一致,参数表示分为几折 是否打乱 设置随机种子
        cv_split().split()  用于生成训练集和验证集的索引
        enumerate() 返回索引和对应的值
        self.data.index为[0,1,2,3,...418]  dataframe类型 索引值
        self.data[f'label_{self.name}'] 为qixu或者xueyu的标签值 dataframe类型 两列左边是索引值右边是标签值
        _表示每折(即每次循环时)作为训练集的那批数据集的索引值
        fold_index表示每折(即每次循环时)作为验证集的那批数据集的索引值
        设置了随机种子 所以每次调用的划分都一致
        """
        for i, (_, fold_index) in enumerate(cv_split.split(self.data.index, self.data[self.name])):  # 遍历每个折叠的索引和对应的数据索引
            if i == test:  # i=0时被划分为测试集索引 剩下的4次都被划分为训练集索引
                test_idx.extend(fold_index)  # 将当前交叉验证折叠的索引fold_index添加到test_idx列表中  .extend()追加   len(test_idx)=84
            else:
                train_idx.extend(fold_index)  # len(train_idx)=335
        
        """
        进一步划分训练集数据,得到训练集和验证集的索引
        train_test_split()函数将训练集划分为训练集和测试集
        self.data.loc[train_idx].index  获取训练集中样本的索引值
        self.data.loc[train_idx][f'label_{self.name}  获取训练集中样本的标签值
        test_size=0.2  指定验证集的大小为整个训练集的20%
        random_state=seed  设置随机种子 确保结果可以复现
        stratify=self.data.loc[train_idx][f'label_{self.name}'   根据标签值进行分层抽样,保证训练集和验证集中各类别样本的比例与原始数据集中的比例一致
        """
        train_idx, val_idx, _, _ = train_test_split(self.data.loc[train_idx].index, self.data.loc[train_idx][self.name], test_size=0.2, random_state=seed, stratify=self.data.loc[train_idx][self.name])
        
        if mode == 'train':  # 选取训练集作为当前数据集  693
            self.data = self.data.loc[train_idx].reset_index(drop=True)  # 选取当前train_idx索引对应的行 并丢弃原来的索引  使得重新设置的索引从0开始连续编号
        elif mode == 'val':  # 174
            self.data = self.data.loc[val_idx].reset_index(drop=True)
        else:
            self.data = self.data.loc[test_idx].reset_index(drop=True)

    # 返回数据集中的样本数量
    def __len__(self):
        return len(self.data)
   
    # 用于实现索引操作
    """
    __getitem__ 接收一个索引index作为参数,用于获取数据集中特定索引位置的样本信息
    sample 创建了一个字典用于存储样本信息
    'id'  就是数据集中的new_id列
    'ask' 数据集中的问诊信息,去除了new_id等列后的剩余列 通过iloc[index]获取到具体的值然后转换为torch.tensor
    'label' 样本的标签,也就是数据集中的label_{self.name}列,转换为整数类型后存储在torch.tensor中
    'img'  根据样本的ID,从图片数据集中获取对应的图片数据,并存储在img键下
    self.data划分好以后 根据self.data中的索引ID tongue_number等选取对应的label img即可
    """
    def __getitem__(self, index):  # 根据传入的索引dataset[0] 可以获取索引位置的相关信息 比如id ask label等
        sample = {
            'id': self.data['tongue_number'][index],
            'ask': torch.tensor(self.data.drop(columns=["ID", "face_number", "tongue_number", "pulse_number", "constitution"]).iloc[index].values, dtype=torch.float32),
            'label': torch.tensor(int(self.data[self.name][index]), dtype=torch.long)
        }
        sample['img'] = self.img[self.data['tongue_number'][index]]  # 将与该样本ID对应的图片数据从图片数据集中取出,并将其存储在字典sample中的'img'键下
        return sample

# 加载数据集
def TCMDataloader(root_path="./data", image_path="", name="", seed=2023, batch_size=32, num_workers=0, k=5, test=0):  # 改动
    train_set = TCMDataset(root_path, image_path, name, mode="train", k=k, test=test, seed=seed)  # 把类实例化成对象
    val_set = TCMDataset(root_path, image_path, name, mode="val", k=k, test=test, seed=seed)
    test_set = TCMDataset(root_path, image_path, name, mode="test", k=k, test=test, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)  #  DataLoader是一个用于批量加载数据的工具,它封装了一个Dataset对象,并提供了对数据的批处理、随机打乱、并行加载等功能
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)  #  DataLoader接收一个Dataset对象作为输入,并允许你指定批量大小、是否打乱数据、采样器等参数
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)  #  通过DataLoader，你可以将数据集封装成一个可迭代的对象,使得在训练模型时可以方便地对数据进行批处理和加载
    return train_loader, val_loader, test_loader

# 模型评估
def evaluate_dev(net, dev_loader, loss, device, f1=False):
    dev_loss_sum, dev_acc_sum, n_samples = 0.0, 0.0, 0
    y_pred, y_true = [], []
    with torch.no_grad():
        net.eval()
        for batch_data in dev_loader:
            b_size = len(batch_data['id'])
            if b_size == 1:
                continue
            x = batch_data['img'].view(b_size, -1).to(device)
            y = batch_data['label']
            y = y.to(device)  # 真实值
            y_hat = net(x)  # 预测值
            l = loss(y_hat, y)
            y_pred.append(y_hat.cpu())
            y_true.append(y.cpu())
            dev_loss_sum += l  # 损失值
            dev_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()  # 准确率
            n_samples += y.shape[0]
    if f1:
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        y_pred_4 = np.argmax(pred, axis=1)
        f1_weighted = f1_score(true, y_pred_4, average='weighted')
        return dev_loss_sum / len(dev_loader), dev_acc_sum / n_samples, f1_weighted
    
    return dev_loss_sum / len(dev_loader), dev_acc_sum / n_samples   # 损失值是每组batch之和除以总batch  准确率是每次预测正确的样本数除以总样本数

# 模型训练
def train_model(net, train_loader, dev_loader, loss, optimizer, n_epochs, device, early_stop=-1, scheduler=None, model_save_path=None):
    net = net.to(device)
    logger.info(f"Training on {device}")
    best_dev_acc, best_epoch = 0.0, 0
    for epoch in range(1, n_epochs+1):
        train_loss_sum, train_acc_sum, n_samples = 0.0, 0.0, 0
        start = time.time()
        net.train()
        for batch_data in tqdm(train_loader):  # batch_data是一个字典 包含 id ask label img 键 
            """
            batch_data是一个字典 包含 id ask label img 键
            x是二维张量[32,768],表示32张图片,每张图片被表示为768个元素的张量
            经过net模型也就是线性模型后得到y_hat
            y_hat是一个二维张量[32,2],表示32张图片,每张图片(也就是每行),被预测称两个类别,每行两个数据取每行最大值的索引值可以理解为就是它最后的预测值
            y是一个一维张量,表示每个图片真实的类别
            通过比较y与y_hat我们可以求得损失值
            """
            b_size = len(batch_data['id'])
            x = batch_data['img'].view(b_size, -1).to(device)  # [32,768]  32张图片 每张图片被表示为一个一维的768个元素的张量
            y = batch_data['label']
            y = y.to(device)  # 真实值(标签) 一个一维张量 存储的是真实的标签值 一个batch(32)个 0 1 张量
            y_hat = net(x)  # 线性层的输出 是一个二维张量 32行 每行两个数据(可近似于理解为概率) 取每行最大值的索引值就是它最后的预测值
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()  # y_hat.argmax(dim=1) 返回每行中最大值的索引值
            n_samples += y.shape[0]
        train_loss = train_loss_sum / len(train_loader)  # 损失值 每组batch的损失相加除以所有batch
        train_acc = train_acc_sum/n_samples  # 每个预测正确的样本除以总样本数

        dev_loss, dev_acc = evaluate_dev(net, dev_loader, loss, device)
        if scheduler is not None:
            scheduler.step(train_loss)

        logger.info("Epoch %d: train_loss: %.4f, train_acc: %.4f, dev_loss: %.4f, dev_acc: %.4f, time: %.1f" % (
            epoch, train_loss, train_acc, dev_loss, dev_acc, time.time()-start))
        
        # 早停策略
        # early_stop设置的是50 即连续50轮 acc的效果都没有提升的话就停止训练
        if early_stop > 0:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch
                if model_save_path is not None:
                    torch.save(net.state_dict(), model_save_path)
            elif epoch - best_epoch >= early_stop:
                logger.info(f"Early Stopped. best_epoch: {best_epoch}. best_dev_acc: {best_dev_acc}\n")
                return
 
    logger.info(f"Finished. best_epoch: {best_epoch}. best_dev_acc: {best_dev_acc}\n")

# 主函数
def main(label_name ="", seed=2023, modality="", path=""):  # 改动
    set_seed(seed)
    num_class = 3
    
    # 实例化线性模型
    net = Model(num_classes=num_class)

    # 配置日志记录器logger 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(Path('./train_log', f"single_tongue_zxh.log"))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    # 参数设置
    batch_size = 32
    n_epochs = 1000
    lr = 0.001
    early_stop = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(net.parameters(), lr=lr)  # Adam损失
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    model_save_path = f"./model_save/single_tongue_zxh.pt"

    logger.info("\n")
    logger.info("==================== Start Training ====================")
    logger.info(f"seed: {seed}")
    logger.info(f"model: {net}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"n_epochs: {n_epochs}")
    logger.info(f"lr: {lr}")
    logger.info(f"early_stop: {early_stop}")
    logger.info(f"loss_fn: {loss_fn}")
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"scheduler: {scheduler}")

    if "after" in path:
        result_name = f'./results/{label_name}_{modality}_after_zxh.csv'
    elif "before" in path:
        result_name = f'./results/{label_name}_{modality}_before_zxh.csv'
    elif "mediapipe" in path:
        result_name = f'./results/{label_name}_{modality}_mediapipe_zxh.csv'
    elif "swin" in path:
        result_name = f'./results/{label_name}_{modality}_swin_zxh.csv'
    else:
        raise ValueError(f"无法识别的路径: {path}")
    
    f = open(result_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['acc', 'f1'])
    k = 5
    for i in range(k):

        # 加载数据集
        train_loader, val_loader, test_loader = TCMDataloader(
            root_path = "./data",
            image_path = path, 
            name = label_name,
            batch_size = batch_size,
            seed=seed,
            k=k,
            test=i
        )

        # 模型训练
        train_model(net, train_loader, val_loader, loss_fn, optimizer, n_epochs, device, early_stop, scheduler, model_save_path)

        net.load_state_dict(torch.load(model_save_path))  # 加载指定模型
        test_loss, test_acc, f1_weighted = evaluate_dev(net, test_loader, loss_fn, device, f1=True)
        logger.info(f"test_loss: {test_loss}, test_acc: {test_acc}, f1_weighted: {f1_weighted}")
        writer.writerow([test_acc, f1_weighted])
    f.close()

# if __name__ == "__main__":
#     argument = [("tongue","/home/zhangxiaohan/code2/constitution_2/data/tongue/feature/feature_vit/tongue.pkl"),
#                 ("tongue_unet","/home/zhangxiaohan/code2/constitution_2/data/tongue/feature/feature_vit_afterseg/tongue_unet.pkl"),]
    
if __name__ == "__main__":
    argument = [("tongue","/home/zhangxiaohan/constitution/tizhi/data/tongue/feature/feature_after/tongue_after_zxh.pkl"),
                ("tongue","/home/zhangxiaohan/constitution/tizhi/data/tongue/feature/feature_before/tongue_before_zxh.pkl"),
                ("tongue","/home/zhangxiaohan/constitution/tizhi/data/tongue/feature/feature_mediapipe/tongue_mediapipe_zxh.pkl"),
                ("tongue","/home/zhangxiaohan/constitution/tizhi/data/tongue/feature/feature_swin/tongue_swin_zxh.pkl")]
    modality, path = argument[3]
    main(label_name="constitution", seed=2023, modality=modality, path=path)