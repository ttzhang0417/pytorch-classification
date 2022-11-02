import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import copy
import pandas as pd
import seaborn as sns
from model import AlexNet
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model(model, traindataloader, train_rate, criterion, optimizer, num_epochs):
    '''
    :param model: 网络模型
    :param traindataloader: 训练数据集，按照train_rate划分为训练集和测试集
    :param tran_rate: 训练集batchsize百分比
    :param criterion: 损失函数
    :param optimizer: 优化方法
    :param num_epochs: 训练轮数
    '''
    # 使用GPU进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model.to(device)
    # 计算训练使用的batch数量
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num*train_rate)   #round函数四舍五入

    train_flag = True
    start_epoch = 0
    # 若train_flag为True，则加载已保存的模型
    save_path = './MyConvNet.pth'
    if train_flag:
        if os.path.isfile(save_path):
            print("Resume from checkpoint...")
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
        else:
            print("====>no checkpoint found.")
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = {}
    train_acc_all = {}
    val_acc_all = {}
    val_loss_all = {}
    since = time.time()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch{}/{}'.format(epoch+1, num_epochs))
        print('-'*10)
        # 每个epoch有两个阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        # 显示训练进度条
        train_bar = tqdm(traindataloader, file=sys.stdout)
        for step, (b_x, b_y) in enumerate(train_bar):
            if step < train_batch_num:
                model.train()   #设置模式为训练模式
                output = model(b_x.to(device))
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y.to(device))
                optimizer.zero_grad()       # # 每个迭代步数的梯度初始化为0
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.to(device).data)
                train_num += b_x.size(0)
            else:
                model.eval()
                output = model(b_x.to(device))
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y.to(device))
                val_loss += loss.item()*b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.to(device).data)
                val_num += b_x.size(0)
        # 计算一个eopch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.doubles().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.doubles().item()/val_num)
        print('train epoch[{}/{}] Train Loss：{:.3f}  Train Acc：{:.3f}'.format(epoch+1, num_epochs, train_loss_all[-1],train_acc_all[-1]))
        print('train epoch[{}/{}] Val Loss：{:.3f}  Val Acc：{:.3f}'.format(epoch+1, num_epochs, val_loss_all[-1], val_acc_all[-1]))
        # 拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = save_path
            torch.save(checkpoint, path_checkpoint)
        time_use = time.time()-since
        print("Train and val complete in {:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))
        # 使用最好模型参数
        model.load_state_dict(best_model_wts)
        train_process = pd.DataFrame(
            data={"epoch": range(start_epoch, num_epochs),
                  "train_loss_all": train_loss_all,
                  "val_loss_all": val_loss_all,
                  "train_acc_all": train_acc_all,
                  "val_acc_all": val_acc_all
                  }
        )
        return model, train_process
def predict_model(validate_loader):
    '''
    :param validate_loader: 测试集
    '''
    # 加载训练过的模型权重文件
    model = AlexNet().to(device)
    weights_path = "./AlexNet.pth"
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 对测试集进行处理
    pre_lab_tensorList = []
    for step, (b_x, b_y) in enumerate(validate_loader):
        model.eval()
        output = model(b_x.to(device))
        pre_lab = torch.argmax(output, 1)
        # tensor->单个python数据，使用data.item()，data必须为tensor且只含有单个数据
        pre_lab_tensorList.append(pre_lab.item())
    # tensor转list，使用tensor.cpu().numpy().tolist()
    test_data_y = validate_dataset.targets
    acc = accuracy_score(test_data_y, pre_lab_tensorList)
    print("在测试集上的预测精度为：{:.3f}".format(acc))
    # 计算混淆矩阵并可视化
    conf_mat = confusion_matrix(test_data_y, pre_lab_tensorList)
    class_label = validate_dataset.classes
    class_label[0] = "apron"
    class_label_index = validate_dataset.class_to_idx
    df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    plt.figure(figsize=(20, 25))
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    # 输出分类报告
    report = classification_report(test_data_y, pre_lab_tensorList, target_names=class_label)
    print(report)
if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据增强与标准化
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(120),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((120, 120)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    image_path = os.path.join(data_root, "dataset")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    print("train_num：", train_num)
    batch_size = 2
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for test.".format(train_num, val_num))
    model = AlexNet()
    # 对模型进行训练
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.00003)
    criterion = nn.CrossEntropyLoss()   # 使用交叉熵损失函数
    model, train_process = train_model(model, train_loader, 0.8, criterion, optimizer, num_epochs=25)
    # 可视化训练结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all,
             "ro-", label="Tain loss")
    plt.plot(train_process.epoch, train_process.val_loss_all,
             "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all,
             "ro-", label="Tain acc")
    plt.plot(train_process.epoch, train_process.val_acc_all,
             "bs-", label="Val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()
    # 预测模型
    predict_model(validate_loader)