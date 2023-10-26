import copy
import os
from network import mnistnet
from torchvision import datasets,transforms
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from util import *
from global_setting import  *
from torch import nn

init_global_model = mnistnet()
global_model_tmp = copy.deepcopy(init_global_model)
# print(init_global_model)
# 导入训练数据
train_dataset = datasets.MNIST(root='./mnist',  # 数据集保存路径
                               train=True,  # 是否作为训练集
                               transform=transforms.ToTensor(),  # 数据如何处理, 可以自己自定义
                               download=True)  # 路径下没有的话, 可以下载

# 导入测试数据
test_dataset = datasets.MNIST(root='./mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)
t = np.arange(6000)
# print(np.choose(t,train_dataset))
samples_per_client = len(train_dataset) // client_N
# print(samples_per_client)
split_index = [samples_per_client]*client_N
# print(split_index)
split_data = torch.utils.data.random_split(dataset=train_dataset,lengths=split_index)
# print((split_data[0][0]))
client_dataloader = []
for i in range(client_N):
    client_dataloader.append(DataLoader(split_data[i]))
print(len(client_dataloader))
client_model = download_model(init_global_model)
# print(client_model)
new_client_model = []
for i in range(client_N):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lr=0.01,params=client_model[i].parameters())
    for e in range(global_epoch):
        for X,y in (client_dataloader[i]):
            # print(X)
            output = client_model[i](X)
            l = loss(output,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    new_client_model.append(client_model[i].state_dict())
    print("="*20+str(i)+" 训练完毕"+"="*20)
global_model = fedavg(new_client_model)
global_model_tmp.load_state_dict(global_model)

correct = 0
total = 0
test_loader = DataLoader(test_dataset)
for images, labels in test_loader:
    outputs = global_model_tmp(images)
    _, predicted = torch.max(outputs, 1)   #返回值和索引
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('测试准确率: {:.4f}'.format(100.0*correct/total))
