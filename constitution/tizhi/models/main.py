import numpy as np
import torch
from sklearn.metrics import *
from torch import nn, optim
from pathlib import Path
from tqdm import tqdm
import time
import logging
import os, sys
import csv
import pandas as pd
import random
import argparse
from model import *  # 导入修改后的模型
from dataloader import TCMDataloader
from unbalanced_loss.focal_loss_classify import MultiClassFocalLossWithAlpha

os.chdir(sys.path[0])  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定GPU，无GPU时自动切换到CPU
logger = logging.getLogger()


def get_arg():
    parser = argparse.ArgumentParser(description='多模态分类参数配置')

    # 数据与路径参数
    parser.add_argument('-imagep', type=str, default='', help="图像数据路径")
    parser.add_argument('-csvp', type=str, default='/home/zhangxiaohan/constitution/data/constitution_threecl_fourmodal.csv', help="数据信息CSV路径")
    parser.add_argument('-logp', type=str, default='/home/zhangxiaohan/constitution/tizhi/models/train_log', help="训练日志保存路径")
    parser.add_argument('-modelp', type=str, default='/home/zhangxiaohan/constitution/tizhi/models/model_save', help="模型保存路径")
    parser.add_argument('-resultp', type=str, default='/home/zhangxiaohan/constitution/tizhi/models/results', help="结果保存路径")

    # 训练参数
    parser.add_argument('-tp', type=float, default=0.8, help="训练集比例")
    parser.add_argument('-bs', type=int, default=32, help="批次大小")
    parser.add_argument('-e', type=int, default=1000, help="训练轮数")
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-es', type=int, default=100, help="早停轮数")
    parser.add_argument('-beta1', type=float, default=0.5, help="Adam优化器beta1")
    parser.add_argument('-beta2', type=float, default=0.999, help="Adam优化器beta2")
    parser.add_argument('-k', type=int, default=5, help="K折交叉验证")
    parser.add_argument('-seed', type=int, default=2023, help="随机种子")

    # 模型与模态参数
    parser.add_argument('-cl', type=str, default='constitution', help="分类标签名称")
    parser.add_argument('-modality', type=str, nargs='+', default=[], help="模态组合 (ask/face/tongue)，空则遍历所有组合")

    return parser.parse_args()


def set_seed(seed):
    """设置随机种子确保可复现性"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_dir(path):
    """创建目录（若不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(log_path, modality, fusion_method):
    """配置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    modality_str = '_'.join(modality)
    log_file = os.path.join(log_path, f"{modality_str}_{fusion_method}_{now_time}.log")
    
    fh = logging.FileHandler(log_file)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    
    return logger


def evaluate_dev(modality, net, dev_loader, loss_fn, device, fusion_type, f1=False):
    """评估函数"""
    dev_loss_sum, dev_acc_sum, n_samples = 0.0, 0.0, 0
    y_pred, y_true = [], []
    
    with torch.no_grad():
        net.eval()
        for batch_data in dev_loader:
            b_size = len(batch_data['id'])
            if b_size == 1:
                continue
            
            x1 = batch_data['ask'].to(device) if 'ask' in modality else None
            x2 = batch_data['face'].to(device) if 'face' in modality else None
            x3 = batch_data['tongue'].to(device) if 'tongue' in modality else None
            y = batch_data['label_name'].to(device)
            
            if fusion_type == "pmdf":
                y_hat = net(x1, x2, x3)
            else:
                y_hat = net(x1, x2, x3)
            
            l = loss_fn(y_hat, y)
            y_pred.append(y_hat.cpu())
            y_true.append(y.cpu())
            dev_loss_sum += l.item()
            dev_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n_samples += y.shape[0]
    
    if f1 and n_samples > 0:
        pred = torch.cat(y_pred)
        true = torch.cat(y_true)
        y_pred_4 = np.argmax(pred.numpy(), axis=1)
        
        loss_result = dev_loss_sum / len(dev_loader)
        acc = accuracy_score(true, y_pred_4)
        pre = precision_score(true, y_pred_4, average='weighted', zero_division=1)
        recall = recall_score(true, y_pred_4, average='weighted', zero_division=1)
        f1 = f1_score(true, y_pred_4, average='weighted', zero_division=1)
        cm = confusion_matrix(true, y_pred_4)
        cr = classification_report(true, y_pred_4, digits=4, output_dict=True, zero_division=1)
        
        return loss_result, acc, pre, recall, f1, cm, cr
    
    return dev_loss_sum / len(dev_loader), dev_acc_sum / n_samples if n_samples > 0 else 0


def train_model(modality, net, train_loader, dev_loader, loss_fn, optimizer, n_epochs, device, 
                fusion_type, early_stop=-1, scheduler=None, model_save_path=None):
    """训练函数"""
    net = net.to(device)
    alignment_loss_fn = CrossModalAlignmentLoss() if fusion_type == "pmdf" else None
    
    logger.info(f"训练设备: {device} | 融合方法: {fusion_type} | 模态组合: {modality}")
    best_dev_acc, best_epoch = 0.0, 0
    
    for epoch in range(1, n_epochs + 1):
        train_loss_sum, train_acc_sum, n_samples = 0.0, 0.0, 0
        start = time.time()
        net.train()
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            b_size = len(batch_data['id'])
            if b_size == 1:
                continue
            
            x1 = batch_data['ask'].to(device) if 'ask' in modality else None
            x2 = batch_data['face'].to(device) if 'face' in modality else None
            x3 = batch_data['tongue'].to(device) if 'tongue' in modality else None
            y = batch_data['label_name'].to(device)
            
            if fusion_type == "pmdf":
                y_hat, _, prompt_feats_list = net(x1, x2, x3, return_weights=True, return_prompt_feats=True)
                cls_loss = loss_fn(y_hat, y)
                align_loss = alignment_loss_fn(prompt_feats_list) if len(prompt_feats_list) > 1 else 0.0
                total_loss = cls_loss + 0.1 * align_loss
            else:
                y_hat = net(x1, x2, x3)
                total_loss = loss_fn(y_hat, y)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss_sum += total_loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n_samples += y.shape[0]
        
        train_loss = train_loss_sum / len(train_loader) if train_loader else 0
        train_acc = train_acc_sum / n_samples if n_samples > 0 else 0
        dev_loss, dev_acc = evaluate_dev(modality, net, dev_loader, loss_fn, device, fusion_type)
        
        if scheduler is not None and train_loss > 0:
            scheduler.step(train_loss)
        
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}, "
            f"time={time.time()-start:.1f}s"
        )
        
        if early_stop > 0:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_epoch = epoch
                if model_save_path is not None:
                    torch.save(net.state_dict(), model_save_path)
            elif epoch - best_epoch >= early_stop:
                logger.info(f"早停触发: 最佳轮次={best_epoch}, 最佳验证准确率={best_dev_acc:.4f}")
                return
    
    logger.info(f"训练结束: 最佳轮次={best_epoch}, 最佳验证准确率={best_dev_acc:.4f}")


def run_experiment(args, fusion_method):
    """运行单个实验"""
    log_path = os.path.join(args.logp, now_day)
    model_path = os.path.join(args.modelp, now_day)
    result_path = os.path.join(args.resultp, now_day)
    create_dir(log_path)
    create_dir(model_path)
    create_dir(result_path)
    
    modality_str = '_'.join(args.modality)
    global logger
    logger = get_logger(log_path, args.modality, fusion_method)
    
    model_save_path = os.path.join(model_path, f"{modality_str}_{fusion_method}_{now_time}.pt")
    
    # 初始化模型
    if fusion_method == "simple_concat":
        net = simple_concat(modality=args.modality)
    elif fusion_method == "simple_concat_attention":
        net = simple_concat_attention(modality=args.modality)
    elif fusion_method == "swin_simple_concat":
        net = swin_simple_concat(modality=args.modality)
    else:
        net = Model(modality=args.modality, fusion=fusion_method)
    
    # 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = MultiClassFocalLossWithAlpha(device=device, gamma=2)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    logger.info("\n==================== 实验配置 ====================")
    logger.info(f"种子: {args.seed}")
    logger.info(f"融合方法: {fusion_method}")
    logger.info(f"模态组合: {args.modality}")
    logger.info(f"批次大小: {args.bs}")
    logger.info(f"训练轮数: {args.e}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"早停轮数: {args.es}")
    logger.info(f"K折交叉验证: {args.k}")
    logger.info(f"损失函数: {loss_fn.__class__.__name__}")
    logger.info(f"优化器: {optimizer}")
    logger.info(f"设备: {device}")
    logger.info("=================================================\n")
    
    # 交叉验证
    result_list = []
    acc_list, pre_list, recall_list, f1_list = [], [], [], []
    cm_df = pd.DataFrame()
    cr_df = pd.DataFrame()
    
    for fold in range(args.k):
        logger.info(f"\n----- 第 {fold+1}/{args.k} 折 -----")
        train_loader, val_loader, test_loader = TCMDataloader(
            root_path=args.imagep,
            csv_path=args.csvp,
            label_name=args.cl,
            modality=args.modality,
            batch_size=args.bs,
            seed=args.seed,
            k=args.k,
            test=fold
        )
        
        train_model(
            modality=args.modality,
            net=net,
            train_loader=train_loader,
            dev_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            n_epochs=args.e,
            device=device,
            fusion_type=fusion_method,
            early_stop=args.es,
            scheduler=scheduler,
            model_save_path=model_save_path
        )
        
        net.load_state_dict(torch.load(model_save_path))
        loss_result, acc, pre, recall, f1, cm, cr = evaluate_dev(
            modality=args.modality,
            net=net,
            dev_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            fusion_type=fusion_method,
            f1=True
        )
        
        logger.info(f"第 {fold+1} 折测试: 损失={loss_result:.4f}, 准确率={acc:.4f}, F1={f1:.4f}")
        result_list.append([str(fold+1), acc, pre, recall, f1])
        acc_list.append(acc)
        pre_list.append(pre)
        recall_list.append(recall)
        f1_list.append(f1)
        
        # cm_df = cm_df.append(pd.DataFrame([[f"第{fold+1}折混淆矩阵", "", ""]]), ignore_index=True)
        # cm_df = cm_df.append(pd.DataFrame(cm), ignore_index=True)
        # cr_df = cr_df.append(pd.DataFrame([[f"第{fold+1}折分类报告", "", ""]]), ignore_index=True)
        # cr_df = cr_df.append(pd.DataFrame(cr).transpose(), ignore_index=True)
    
    avg_acc = np.mean(acc_list)
    avg_pre = np.mean(pre_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    result_list.append(["平均值", avg_acc, avg_pre, avg_recall, avg_f1])
    logger.info(f"\n===== 交叉验证平均结果 =====")
    logger.info(f"平均准确率: {avg_acc:.4f}, 平均精确率: {avg_pre:.4f}")
    logger.info(f"平均召回率: {avg_recall:.4f}, 平均F1: {avg_f1:.4f}")
    
    # 保存结果
    result_file = os.path.join(result_path, f"{args.cl}_{modality_str}_{fusion_method}_{now_time}_results.csv")
    result_header = ['K折', '准确率', '精确率', '召回率', 'F1分数']
    pd.DataFrame(result_list).to_csv(
        result_file, encoding='utf_8_sig', index=False, header=result_header
    )
    
    # cm_file = os.path.join(result_path, f"{args.cl}_{modality_str}_{fusion_method}_{now_time}_confusion_matrix.csv")
    # cm_df.to_csv(cm_file, encoding='utf_8_sig', index=False, header=False)
    
    # cr_file = os.path.join(result_path, f"{args.cl}_{modality_str}_{fusion_method}_{now_time}_classification_report.csv")
    # cr_df.to_csv(cr_file, encoding='utf_8_sig', index=False)


def main():
    global now_time, now_day
    now_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    now_day = time.strftime('%Y%m%d', time.localtime())
    
    args = get_arg()
    set_seed(args.seed)
    
    # 配置要实验的融合方法
    fusion_methods = [
        "pmdf"          # PMDF融合（最新）
        # "lr",             # 低秩张量融合
        # "tr",             # Transformer融合
        # "simple_concat",  # 简单拼接
        # "simple_concat_attention"  # 带注意力的拼接
    ]
    
    # 配置要实验的模态组合
    if not args.modality:
        all_modalities = [
            ['ask'], ['face'], ['tongue'],
            ['ask', 'face'], ['ask', 'tongue'], ['face', 'tongue'],
            ['ask', 'face', 'tongue']
        ]
    else:
        all_modalities = [args.modality]
    
    # 遍历所有组合
    for fusion_method in fusion_methods:
        for modality in all_modalities:
            args.modality = modality
            logger.info(f"\n\n==================== 开始实验: {fusion_method} + {modality} ====================")
            run_experiment(args, fusion_method)


if __name__ == "__main__":
    main()