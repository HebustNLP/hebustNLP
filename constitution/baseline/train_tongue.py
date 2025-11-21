import argparse
import os.path
from datetime import datetime
import time
import shutil

from prettytable import PrettyTable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from sklearn.metrics import *

from dataload_tongue import get_dataloader
from model.resnet import *
from model.VGG import *
from model.DenseNet import *
# from try.grad_cam.vit_model import vit_base_patch16_224_in21k

from utils.create_log import logger_get
# from data.scatter.scatter_prepare import Classification_scatter_data
from utils.report_utils import df_generator, torch2onnx
from utils.draw_utils import *

from transformers import ViTForImageClassification

from vit_model import *

# 导入参数
def get_arg():
    parser = argparse.ArgumentParser(description='classification parameter configuration(train)')
    parser.add_argument(
        '-t',
        type=str,
        default='pytorch-imageclassification-master',
        help='This is your task theme name'
    )
    parser.add_argument(
        '-imagep',
        type=str,
        default="/home/zhangxiaohan/constitution/data/face",
        help="image's directory"
    )
    parser.add_argument(
        '-csvp',
        type=str,
        default="/home/zhangxiaohan/constitution/baseline/data.csv",
        help="DIF(data information file)'s path"
    )
    parser.add_argument(
        '-tp',
        type=float,
        default=0.8,
        help="train data's percent"
    )
    parser.add_argument(
        '-bs',
        type=int,
        default=16,
        help="train data's batch size"
    )
    parser.add_argument(
        '-rs',
        type=tuple,
        default=(224, 224),
        help='resized shape of input tensor'
    )
    # parser.add_argument(
    #     '-clsp',
    #     type=str,
    #     default=r'D:\guan\guanclassification\dataset_0409_3\classes_guan.txt',
    #     help="classes.txt's path"
    # )
    parser.add_argument(
        '-cn',
        type=int,
        default=2,
        help='the number of classes'
    )
    parser.add_argument(
        '-e',
        type=int,
        default=300,
        help='epoch'
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '-beta1',
        type=float,
        default=0.5,
        help='beta1 of Adam'
    )
    parser.add_argument(
        '-beta2',
        type=float,
        default=0.999,
        help='beta2 of Adam'
    )
    parser.add_argument(
        '-ld',
        type=str,
        default='/home/zhangxiaohan/constitution/baseline/workdir',
        help="the training log's save directory"
    )

    parser.add_argument(
        '-model',
        type=str,
        default='vit1621k_0725',
        help="resnet18/ resnet34/resnet50/ "
             "vgg11/ vgg13/ vgg16/ vgg19"
             "densenet121/ densenet169/ densenet201/ densenet161"
    )

    parser.add_argument(
        '-cl',
        type=str,
        default='constitution',
        help="the training class label: tongue_color/crack/tooth/shape/coating_color/coating_thickness"
    )

    return parser.parse_args()

# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 任务开始前的准备工作
    args = get_arg()  # 得到参数Namespace
    a = args.ld
    if not os.path.exists(args.ld):
        os.makedirs(args.ld)
    # nowtime = datetime.now()    # 获取任务开始时间
    now_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = "{}/exp_{}_{}_{}".format(args.ld,args.cl,args.model,now_time)
    os.makedirs(log_dir,exist_ok=True)
    log_path = os.path.join(log_dir,"exp_{}_{}_{}.log".format(args.cl,args.model,now_time))
    file_logger = logger_get(log_path)  # 获取logger
    # source_data = Classification_scatter_data(args.imagep,args.clsp,args.csvp)
    file_logger.info("Read csv file.......")
    # try:
    #     source_data.generate()
    #     file_logger.info("Done.")
    # except:
    #     file_logger.error("Generate failure!")
    # 训练设备信息
    device_table = ""
    if torch.cuda.is_available():
        device_table = PrettyTable(['number of gpu', 'applied gpu index', 'applied gpu name'], min_table_width=80)
        gpu_num = torch.cuda.device_count()
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name()
        device_table.add_row([str(gpu_num), str(gpu_index), str(gpu_name)])
        file_logger.info('Training device information:\n{}\n'.format(device_table))
    else:
        file_logger.warning("Using cpu......")
        device_table = 'CPU'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------------------------------------------
    # 数据集信息
    file_logger.info("Use data information file:{}\nLoading data from path: {}......".format(args.csvp, args.imagep))
    train_dl, valid_dl, test_dl, samples_num, train_num, valid_num, test_num = get_dataloader(args.imagep, args.csvp, args.rs,
                                                                                              args.cl,args.bs, args.tp)
    dataset_table = PrettyTable(['number of samples', 'train number', 'valid number','test number', 'percent'], min_table_width=80)
    dataset_table.add_row([samples_num, train_num, valid_num,test_num, args.tp])
    file_logger.info("data information:\n{}\n".format(dataset_table))
# ----------------------------------------------------------------------------------------------------------------------
    # 类别信息
    # classes = source_data.cls_txt2buffer()
    classes = list(range(args.cn))
    # classes = [0,1,3,4,5,6,7]
    classes_table = PrettyTable(classes, min_table_width=80)
    classes_table.add_row(range(len(classes)))
    # classes_table.add_row(classes)
    file_logger.info("Classes information:\n{}\n".format(classes_table))
# ----------------------------------------------------------------------------------------------------------------------
    # 训练组件配置
    if args.model == 'resnet18':
        model = resnet18(num_classes=args.cn).to(device)
    elif args.model == 'resnet34':
        model = resnet34(num_classes=args.cn).to(device)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=args.cn).to(device)
    elif args.model == 'vgg11':
        model = vgg(num_classes=args.cn, model_name="vgg11").to(device)
    elif args.model == 'vgg13':
        model = vgg(num_classes=args.cn, model_name="vgg13").to(device)
    elif args.model == 'vgg16':
        model = vgg(num_classes=args.cn, model_name="vgg16").to(device)
    elif args.model == 'vgg19':
        model = vgg(num_classes=args.cn, model_name="vgg19").to(device)
    elif args.model == 'densenet121':
        model = densenet121(num_classes=args.cn).to(device)
    elif args.model == 'densenet169':
        model = densenet169(num_classes=args.cn).to(device)
    elif args.model == 'densenet201':
        model = densenet201(num_classes=args.cn).to(device)
    elif args.model == 'densenet161':
        model = densenet161(num_classes=args.cn).to(device)
    elif args.model == 'vit16':
        model =ViTForImageClassification.from_pretrained('/home/zhangxiaohan/constitution/baseline/vit-base-patch16-224', num_labels=2,ignore_mismatched_sizes=True).to(device)
    elif args.model == 'vit16_0725':
        model = vit_base_patch16_224(num_classes=2).to(device)
    # elif args.model == 'vit16pretrain_0725':
    #     model = vit_base_patch16_224(num_classes=2).to(device)
    #     model.load_state_dict(torch.load('/home/liuhuilin/try/grad_cam/vit_base_patch16_224.pth'))
    elif args.model == 'vit1621k_0725':
        model = vit_base_patch16_224_in21k(num_classes=2).to(device)


    # optimizer = Adam(params=model.parameters(), lr=args.lr)
    optimizer = Adam(params=model.parameters(), lr=args.lr, betas=(args.beta1,args.beta1))
    loss_fn = CrossEntropyLoss()    ##################################################
    train_table = PrettyTable(['theme', 'resize', 'batch size', 'epoch', 'learning rate','beta1','beta2', 'directory of log'],
                              min_table_width=180)
    train_table.add_row([args.cl, args.rs, args.bs, args.e, args.lr,args.beta1, args.beta2,  args.ld])
    file_logger.info('Train information:\n{}\n'.format(train_table))
# ----------------------------------------------------------------------------------------------------------------------
    # 开始训练
    file_logger.info("Train begins......")
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    best_checkpoint = 0.
    shapeuse = 0
    preduse = 0
    labeluse = 0
    st = datetime.now()

    for epoch in range(args.e):

        prediction = []
        label = []
        score = []

        model.train(True)
        train_bar = tqdm(iter(train_dl), ncols=150, colour='blue')
        train_loss = 0.
        i = 0
        for train_data in train_bar:
            x_train, y_train = train_data
            shapeuse = x_train.shape
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            output = model(x_train)
            # output = model(x_train).logits
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            # clone().detach()：可以仅仅复制一个tensor的数值而不影响tensor# 原内存和计算图
            train_loss += loss.clone().detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            # 显示每一批次的loss
            train_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(train_dl)))
            train_bar.set_postfix({"train loss": "%.3f" % loss.data})
            i += 1
        train_loss = train_loss / i
        file_logger.info("Epoch loss:{}".format(train_loss))
        # 最后得到的i是一次迭代中的样本数批数
        losses.append(train_loss)

        model.train(False)
        model.eval()
        valid_bar = tqdm(iter(valid_dl), ncols=150, colour='blue')
        valid_acc = 0.
        valid_pre = 0.
        valid_recall = 0.
        valid_f1 = 0.
        i = 0
        for valid_data in valid_bar:
            x_valid, y_valid = valid_data
            x_valid = x_valid.to(device)
            y_valid_ = y_valid.clone().detach().numpy().tolist()  # y_valid就不必放到gpu上训练了
            output = model(x_valid)  # shape:(N*cls_n)
            output_ = output.clone().detach().cpu()
            _, pred = torch.max(output_, 1)  # 输出每一行(样本)的最大概率的下标
            pred_ = pred.clone().detach().numpy().tolist()
            output_ = output_.numpy().tolist()
            # 显示每一批次的acc/precision/recall/f1
            valid_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(valid_dl)))
            prediction = prediction + pred_
            label = label + y_valid_
            score = score + output_
            i += 1
        # 最后得到的i是一次迭代中的样本数批数,每一次epoch计算一次indicators
        valid_acc = accuracy_score(y_true=label, y_pred=prediction)
        valid_pre = precision_score(y_true=label, y_pred=prediction, average='weighted')
        valid_recall = recall_score(y_true=label, y_pred=prediction, average='weighted')
        valid_f1 = f1_score(y_true=label, y_pred=prediction, average='weighted')
        preduse = prediction
        labeluse = label
        # 验证阶段信息输出
        indicator_table = PrettyTable(['Accuracy', 'Precision', 'Recall', 'F1'], )
        indicator_table.add_row([valid_acc, valid_pre, valid_recall, valid_f1])
        file_logger.info('\n{}\n'.format(indicator_table))
        # indicator保存
        accuracies.append(valid_acc)
        precisions.append(valid_pre)
        recalls.append(valid_recall)
        f1s.append(valid_f1)
        # 保存最好的f1指标的checkpoint
        if valid_f1 >= max(f1s):    # 如果本次epoch的f1大于了存储f1列表的最大值，那么最好的checkpoint赋值为model
            best_checkpoint = model
        # 保存每次的checkpoint，从而实现断点继训
        os.makedirs("/home/zhangxiaohan/constitution/baseline/checkpoint",exist_ok=True)  # 项目根路径下的checkpoints目录下保存临时checkpoint
        if not os.path.exists("/home/zhangxiaohan/constitution/baseline/checkpoint/train_info.txt"):
            with open("/home/zhangxiaohan/constitution/baseline/checkpoint/info.txt",'w') as f:
                content = "{}\n{}\n{}\n{}y\n{}\n".format(dataset_table,classes_table,device_table,train_table,optimizer)
                f.write(content)
        # torch.save(model, "/home/liuhuilin/psy/code/checkpoints/{}.pth".format(epoch))
    et = datetime.now()
    # 训练完，记得把model和优化器也加入到日志中（训练完加入以防训练前对model或者优化器产生影响）
    file_logger.info("optimizer:\n{}\nmodel:\n{}\n".format(optimizer, model))
# ----------------------------------------------------------------------------------------------------------------------
    # 完成训练后的断电续训的临时文件的删除、日志保存(程序结束后自动保存)以及绘图等后续工作
    # 删除临时checkpoints文件以及临时信息文件
    # cmd = input("是否删除临时文件和临时信息文件？[y/n]")
    cmd = 'y'
    if cmd == 'y':
        shutil.rmtree("/home/zhangxiaohan/constitution/baseline/checkpoint")
    # indicators和loss的df记录文件生成
    df = df_generator(args.e, [losses, accuracies, precisions, recalls, f1s], os.path.join(log_dir, 'indicators.csv'))
    # 权重生成(onnx/pth + bestf1/last)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model, os.path.join(checkpoint_dir, 'last.pth'))  # 最后一次的checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'last_weight.pth'))
    # try:
    #     torch2onnx(os.path.join(checkpoint_dir, 'last.pth'), os.path.join(checkpoint_dir, 'last.onnx'), shapeuse)
    #     file_logger.info("Last model transforms successfully and path is {}.".format(os.path.join(checkpoint_dir, 'last.onnx')))
    # except:
    #     file_logger.warning("Last model transforms failed.")

    torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_f1.pth'))  # 最好的f1的checkpoint
    torch.save(best_checkpoint.state_dict(), os.path.join(checkpoint_dir, 'best_f1_weight.pth'))
    # try:
    #     torch2onnx(os.path.join(checkpoint_dir, 'best_f1.pth'), os.path.join(checkpoint_dir, 'best_f1.onnx'), shapeuse)
    #     file_logger.info("Best model transforms successfully and path is {}.".format(os.path.join(checkpoint_dir, 'best_f1.onnx')))
    # except:
    #     file_logger.warning("Best model transforms failed.")
# ----------------------------------------------------------------------------------------------------------------------

    # 绘图（当然也可以选择使用提供的函数在训练后绘制，一些参数可以在宝库函数中自行调整）
    # 1.绘制loss和indicators变化曲线
    log_plot(df,log_dir)
    # 2.绘制数据分布图
    dataset_distribution(args.csvp,classes,args.cl,log_dir)
    # 3.用最好的f1的checkpoints绘制中间特征图(测试图片可以自己选)
    # generate_feature('201509070028.jpg', args.rs, os.path.join(checkpoint_dir, 'best_f1.pth'),os.path.join(log_dir, 'feature_maps'))
    # 4.绘制最后一次的热力图——当然可以根据自己改预测和标签
    # get_confusion_matrix(y_pred=preduse, y_label=labeluse, cls_num=args.cn, fig_save_dir=log_dir)


# -----------------------------------------------------------------------TEST-----------------------------------------------
    test_model = torch.load(os.path.join(checkpoint_dir, 'best_f1.pth'),weights_only=False)
    model.train(False)
    model.eval()
    test_prediction = []
    test_label = []
    test_score = []
    test_acc = 0.
    test_pre = 0.
    test_recall = 0.
    test_f1 = 0.
    for i, (x_test, y_test) in enumerate(test_dl.dataset):
        # x_test, y_test = test_data
        x_test = x_test.to(device)
        x_test_ = x_test.reshape((1, *x_test.shape))
        y_test_ = [y_test] # y_valid就不必放到gpu上训练了
        output = model(x_test_)  # shape:(N*cls_n)
        output_ = output.clone().detach().cpu()
        _, pred = torch.max(output_, 1)  # 输出每一行(样本)的最大概率的下标
        pred_ = pred.clone().detach().numpy().tolist()
        output_ = output_.numpy().tolist()
        # 显示每一批次的acc/precision/recall/f1
        # test_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(valid_dl)))
        test_prediction = test_prediction + pred_
        test_label = test_label + y_test_
        test_score = test_score + output_
        # i += 1
    # 最后得到的i是一次迭代中的样本数批数,每一次epoch计算一次indicators
    test_acc = accuracy_score(y_true=test_label, y_pred=test_prediction)
    test_pre = precision_score(y_true=test_label, y_pred=test_prediction, average='weighted')
    test_recall = recall_score(y_true=test_label, y_pred=test_prediction, average='weighted')
    test_f1 = f1_score(y_true=test_label, y_pred=test_prediction, average='weighted')
    test_reports = classification_report(test_label, test_prediction, digits=4)
    test_confusion = confusion_matrix(test_label, test_prediction)
    preduse = test_prediction
    labeluse = test_label
    # 测试阶段信息输出

    # indicator_table = PrettyTable(['Accuracy', 'Precision', 'Recall', 'F1'], )
    # indicator_table.add_row([test_acc, test_pre, test_recall, test_f1])
    file_logger.info("TEST")
    file_logger.info('\n{}\n'.format(indicator_table))
    file_logger.info('\n{}\n'.format(test_reports))
    file_logger.info('\n{}\n'.format(test_confusion))










