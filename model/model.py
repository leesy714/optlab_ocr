import os
import json
import fire
import cv2
import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging

from focal_loss import FocalLoss


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def accuracy(y_true, y_pred):
    total = y_true.size(0)
    correct = (y_true == y_pred).sum().item()
    return correct / (total + 1e-8)

def recall(y_true, y_pred):
    positive = (y_true > 0)
    total = positive.sum().item()
    correct = (positive[y_true == y_pred]).sum().item()
    return correct / (total + 1e-8)

def precision(y_true, y_pred):
    positive = (y_pred > 0)
    total = positive.sum().item()
    correct = (positive[y_true == y_pred]).sum().item()
    return correct / (total + 1e-8)

def accuracy_bbox(y_true, y_pred, bboxes):
    acc = []
    batch = y_true.size(0)
    for b in range(batch):
        _y_true = y_true[b]
        _y_pred = y_pred[b]
        _bbox = bboxes[b]
        for box in _bbox:
            label, x1, y1, x2, y2, x3, y3, x4, y4 = box
            x_min, x_max = min(x1, x2, x3, x4)//2, max(x1, x2, x3, x4)//2
            y_min, y_max = min(y1, y2, y3, y4)//2, max(y1, y2, y3, y4)//2
            if x_min == x_max or y_min == y_max:
                continue
            y_pred_box = torch.flatten(_y_pred[y_min:y_max, x_min:x_max])
            uniq2, counts = torch.unique(y_pred_box, return_counts=True)
            acc.append(label == uniq2[0])
    return sum(acc) / (len(acc) + 1e-8)

class Data(Dataset):

    def __init__(self):
        #self.classes = classes
        #x, y, base = self.load()
        #print("data shape", x.shape, y.shape, base.shape)
        #self.x = x
        #self.y = y
        #self.base = base
        self.base_dir = "../data/imgs/origin_noise"
        self.ipt_dir = "../data/imgs/origin_craft"
        self.opt_dir = "../data/imgs/origin_noise_label"
        assert len(os.listdir(self.ipt_dir)) == len(os.listdir(self.opt_dir)) == len(os.listdir(self.base_dir))

        self.data_len = len(os.listdir(self.ipt_dir))

    def __getitem__(self, idx):
        base = cv2.imread(os.path.join(self.base_dir, str(idx)+".jpg"))
        x = np.load(os.path.join(self.ipt_dir, str(idx)+".npy"))
        y = np.load(os.path.join(self.opt_dir, str(idx)+".npy"))
        x = x.reshape(640, 480, 1)
        x = x.transpose(2, 0, 1)
        y = y.reshape(640, 480)
        base = base.reshape(1280, 960, 3)
        return x, y.astype(np.int64), base, idx

    def __len__(self):
        return self.data_len


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=(9, 7), padding=(4, 3)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=(9, 7), padding=(4, 3)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Model(nn.Module):

    def __init__(self, classes):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(9, 7), padding=(4, 3)),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.conv1 = double_conv(4, 8, 16)
        self.conv2 = double_conv(16, 32, 64)


        self.conv_cls = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(16, classes + 1, kernel_size=1),
        )

        init_weights(self.conv.modules())
        init_weights(self.conv1.modules())
        init_weights(self.conv2.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        y = self.conv(x)
        feature = self.conv1(y)
        feature = self.conv2(feature)
        y = self.conv_cls(feature)

        return y, feature

class Train:

    def __init__(self, classes, batch_size=4, loss_gamma=0, loss_alpha=0.5, epochs=10):
        self.classes = classes
        self.epochs =  epochs
        self.learning_rate = 0.005
        self.data = Data()
        self.loss_gamma = loss_gamma
        self.loss_alpha = loss_alpha

        data_len = len(self.data)
        self.train_num = int(data_len * 0.8)
        self.vali_num = int(data_len * 0.1)
        self.test_num = data_len - self.train_num - self.vali_num
        train, vali, test = random_split(self.data, [self.train_num, self.vali_num, self.test_num])
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
        self.vali_loader = DataLoader(vali, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

        self.model = Model(self.classes).to(device)


    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))

        for batch, (imgs, ys, _, idx) in pbar:
            imgs = imgs.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()
            pred, _ = self.model(imgs)
            loss = loss_func(pred, ys)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= (self.train_num)
        return total_loss[0]

    def validate(self, epoch, iterator, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])

        accs, recs, pres, acc_boxes = [] ,[], [], []
        for batch, (imgs, ys, _, file_num) in enumerate(iterator):
            imgs = imgs.to(device)
            ys = ys.to(device)
            pred, _ = self.model(imgs)
            loss = loss_func(pred, ys)
            total_loss += loss.item()
            pred = pred.permute(0, 2, 3, 1)
            _, argmax = pred.max(dim=3)
            bbox = [self.load_bbox(b) for b in file_num]
            acc_box = accuracy_bbox(ys, argmax, bbox)
            acc = accuracy(ys.view(-1, 1), argmax.view(-1, 1))
            rec = recall(ys.view(-1, 1), argmax.view(-1, 1))
            pre = precision(ys.view(-1, 1), argmax.view(-1, 1))
            accs.append(acc)
            acc_boxes.append(acc_box)
            recs.append(rec)
            pres.append(pre)
        acc = sum(accs) / len(accs)
        acc_box = sum(acc_boxes) / len(acc_boxes)
        rec = sum(recs) / len(recs)
        pre = sum(pres) / len(pres)
        print("acc: {}, recall: {}, precision: {}, acc_bbox: {}".format(acc, rec, pre, acc_box))

        total_loss /= (self.vali_num)
        return total_loss[0], acc, rec, pre, acc_box

    def load_bbox(self, file_num):
        with open("../data/imgs/origin_noise_bbox/" + str(int(file_num)) + ".pickle", "rb") as fin:
            bbox = pickle.load(fin)
        return bbox

    def test(self):
        self.model.eval()

        for batch, (imgs, ys, bases, file_num) in enumerate(self.test_loader):
            imgs = imgs.to(device)
            pred, _ = self.model(imgs)
            pred = pred.permute(0, 2, 3, 1)
            _, argmax = pred.max(dim=3)
            argmax = argmax.cpu().data.numpy()
            imgs = imgs.squeeze().cpu().data.numpy()

            bases = bases.squeeze().cpu().data.numpy()
            for idx in range(argmax.shape[0]):
                bbox = self.load_bbox(int(file_num[idx]))
                img = argmax[idx]
                origin = np.clip(imgs[idx] * 255, 0, 255).astype(np.uint8)
                img = np.clip(img*(255 / self.classes), 0, 255).astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                origin = cv2.applyColorMap(origin, cv2.COLORMAP_JET)
                cv2.imwrite("./res/{}_base.png".format(idx), bases[idx])
                cv2.imwrite("./res/{}.png".format(idx), img)
                cv2.imwrite("./res/{}_craft.png".format(idx), origin)
            break

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, res):
        with open("res/{}-{}.json".format(self.loss_gamma, self.loss_alpha), "w") as fout:
            json.dump(res, fout)

    def run(self):
        #loss_func = nn.CrossEntropyLoss()
        alpha = [1-self.loss_alpha for _ in range(self.classes + 1)]
        alpha[0] = self.loss_alpha
        loss_func = FocalLoss(gamma=self.loss_gamma, alpha=alpha)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=0)
        print(self.model)
        print("parameters: ",self.count_parameters())
        res = {}
        tot_vali_loss = np.inf
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, loss_func, optimizer)
            vali_loss, acc, rec, pre, acc_box = self.validate(epoch, self.vali_loader, loss_func)
            res[epoch] = dict(train_loss=float(train_loss), vali_loss=float(vali_loss),
                              accuracy=float(acc), recall=float(rec), precision=float(pre),
                              accuracy_bbox=float(acc_box))
            print("train loss: {:.4} vali loss: {:.4}, rec: {:.4}, box_acc: {:.4}".format(train_loss, vali_loss, rec, acc_box))
        if not os.path.exists("weight"):
            os.makedirs("weight")
        self.test()
        self.save(res)
        torch.save(self.model.cpu().state_dict(), os.path.join("weight", "fully_connected.pt"))

if __name__ == "__main__":
    train = Train(classes=9, epochs=50, batch_size=8, loss_gamma=0.0, loss_alpha=0.25)
    train.run()

