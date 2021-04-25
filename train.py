import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Normalize
from paddle.vision.models import LeNet
import math
import numpy as np
import random
import os
from PIL import Image
normalize = Normalize(mean=[127.5],
                        std=[127.5],
                        data_format='HWC')


#创建数据集读取
class RandomDataset(Dataset):
    def __init__(self, root = "data",mode="train"):
        self.mode = mode
        if mode == "train":
            self.txt_file = os.path.join(root,"train.txt")
        else:
            self.txt_file = os.path.join(root,"test.txt")
        self.records_list = []
        self.parse_dataset()

    def __getitem__(self, idx):
        path=self.records_list[idx][0]
        img = Image.open(path)
        img = img.resize((28, 28), Image.BILINEAR)
        img = normalize(img)
        label = int(self.records_list[idx][1])
        return img, label

    def __len__(self):
        return len(self.records_list)
    
    def parse_dataset(self):
        print(self.txt_file)
        with open(self.txt_file,'r') as f:
            for line in f.readlines():
                img_path, label = line.strip().split()
                self.records_list.append([img_path,label])
        random.shuffle(self.records_list)
train_dataset=RandomDataset()
test_dataset=RandomDataset(mode="test")


train_dataset=RandomDataset()
test_dataset=RandomDataset(mode="test")
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=32, shuffle=False)
# 加载训练集 batch_size 设为 64
def train(model):
    model.train()
    epochs = 10
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            y_data = paddle.unsqueeze(y_data, axis=1)
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data,k=1)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()
        model.eval()
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            y_data = paddle.unsqueeze(y_data, axis=1)
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            test_acc = paddle.metric.accuracy(predicts, y_data,k=1)
            #loss.backward()
        print("test:loss is: {}, acc is: {}".format(loss.numpy(), acc.numpy()))
        model.train()
        if epoch % 2 == 0:
            paddle.save(model.state_dict(), "Tibetan_epoch"+str(epoch)+".pdparams")
model = LeNet()
train(model)