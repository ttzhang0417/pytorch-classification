import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import copy
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(120),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((120, 120)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "dataset")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    print("train_num：",train_num)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}

    image_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in image_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))
    # 启动模型，将模型放入GPU
    net = AlexNet()
    net.to(device)
    # 设置训练需要的参数，epoch、学习率、优化器、损失函数
    loss_function = nn.CrossEntropyLoss()           # 交叉熵损失，pytorch中常用多分类问题的损失函数
    learning = 0.00001
    optimizer = optim.Adam(net.parameters(), lr=learning)
    epochs = 120
    save_path = './AlexNet.pth'
    # 复制模型的参数
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    train_steps = len(train_loader)
    # 设置四个空数组，用来存放训练集的loss和accuracy，测试集的loss和accuracy
    train_loss_all = []
    train_accur_all = []
    test_loss_all = []
    test_accur_all = []
    # 开始训练
    train_flag = True
    start_epoch = 0
    # 若train_flag为True，则加载已保存的模型
    if train_flag:
        if os.path.isfile(save_path):
            print("Resume from checkpoint...")
            checkpoint = torch.load(save_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']+1
            print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
        else:
            print("====>no checkpoint found.")
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
    for epoch in range(start_epoch, epochs):
        train_loss = 0.0
        train_num = 0.0
        train_accuracy = 0.0
        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()   # 每个迭代步数的梯度初始化为0
            outputs = net(images.to(device))
            loss1 = loss_function(outputs, labels.to(device))
            outputs = torch.argmax(outputs, 1)
            loss1.backward()        # 损失的反向传播，计算梯度
            optimizer.step()        # 使用梯度进行优化
            # train_loss_steps.append(loss1.item())
            # 计算每经过print_step次迭代后的输出
            train_loss += abs(loss1.item()*images.size(0))
            accuracy = torch.sum(outputs == labels.to(device))
            train_accuracy = train_accuracy + accuracy
            train_num  += images.size(0)
        print("train epoch[{}/{}] train-loss:{:.3f} train-accuracy:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     train_loss/train_num, train_accuracy/train_num))
        train_loss_all.append(train_loss/train_num)
        train_accur_all.append(train_accuracy/train_num)
        #开始测试
        test_loss = 0.0
        test_accuracy = 0.0
        test_num = 0
        net.eval()
        with torch.no_grad():       # 禁止梯度计算
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss2 = loss_function(outputs, val_labels.to(device))
                outputs = torch.argmax(outputs, 1)
                # test_loss_steps.append(loss2.item())
                test_loss = test_loss+abs(loss2.item()*images.size(0))
                accuracy = torch.sum(outputs == val_labels.to(device))
                test_accuracy = test_accuracy + accuracy
                test_num += images.size(0)
        print('[epoch %d] test_loss: %.3f  test_accuracy: %.3f'%
              (epoch + 1, test_loss / test_num, test_accuracy/test_num))
        test_loss_all.append(test_loss/test_num)
        test_accur_all.append(test_accuracy/test_num)
        # 拷贝模型最高精度下的参数
        if test_accur_all[-1] > best_acc:
            best_acc = test_accur_all[-1]
            # best_model_wts = copy.deepcopy(net.state_dict())
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = save_path
            torch.save(checkpoint, path_checkpoint)
    # 可视化loss和acc
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(start_epoch, epochs), train_loss_all,
             "ro-", label="Train loss")
    plt.plot(range(start_epoch, epochs), test_loss_all,
             "bs-", label="test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(start_epoch, epochs), train_accur_all,
             "ro-", label="Train accur")
    plt.plot(range(start_epoch, epochs), test_accur_all,
             "bs-", label="test accur")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
    # # 使用最好模型的参数
    # net.load_state_dict(best_model_wts)
    # torch.save(net.state_dict(), save_path)
    print("训练结束，模型已保存")


if __name__ == '__main__':
    main()
