import os
import json
import fire
import cv2
import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import logging

from data import Data
from model import Model3 as Model
from focal_loss import FocalLoss
from optimizer import RAdam

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'


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
            uniq, counts = torch.unique(y_pred_box, return_counts=True)
            uniq, counts = enumerate(list(uniq.cpu().numpy())), list(counts.cpu().numpy())
            uniq = sorted(uniq, key=lambda v: counts[v[0]], reverse=True)
            uniq = [u[1] for u in uniq]
            value = uniq[0] if uniq[0] != 0 or len(uniq) <= 1 else uniq[1]
            acc.append(value == label)
    return sum(acc) / (len(acc) + 1e-8)


class Train:

    def __init__(self, classes=9, batch_size=8, loss_gamma=0.1, loss_alpha=0.4,
                 learning_rate=1e-4, epochs=10):
        """
        :param classes: number of classes except background
        :param batch_size: training dataset batch size
        :param loss_gamma: gamma value in focal loss
        :param loss_alpha: alpha value in focal loss
        :param learning_rate: Optimizer learning_rate
        :param epochs: training_epochs
        """

        self.classes = classes
        self.epochs =  epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_gamma = loss_gamma
        self.loss_alpha = loss_alpha

        self.data = Data()
        data_len = len(self.data)
        self.train_num = int(data_len * 0.8)
        self.vali_num = int(data_len * 0.1)
        self.test_num = data_len - self.train_num - self.vali_num
        train = Subset(self.data, list(range(self.train_num)))
        vali = Subset(self.data, list(range(self.train_num, self.train_num + self.vali_num)))
        test = Subset(self.data, list(range(self.train_num + self.vali_num, data_len)))
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
        self.vali_loader = DataLoader(vali, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

        self.model = Model(self.classes)
        self.model_name = self.model.__class__.__name__

        if torch.cuda.device_count()>1:
            print("Use ", torch.cuda.device_count(), 'GPUs')
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device)


    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))

        for batch, (crafts, ys, imgs, idx) in pbar:
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.to(device)
            ys = ys.to(device)
            crafts = crafts.to(device)
            x = torch.cat((imgs, crafts), dim=1)
            optimizer.zero_grad()
            pred, _ = self.model(x)
            loss = loss_func(pred, ys)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= len(self.train_loader)
        return total_loss[0]

    def validate(self, epoch, iterator, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])

        accs, recs, pres, acc_boxes = [] ,[], [], []
        for batch, (crafts, ys, imgs, file_num) in enumerate(iterator):
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.to(device)
            ys = ys.to(device)
            crafts = crafts.to(device)
            x = torch.cat((imgs, crafts), dim=1)
            pred, _ = self.model(x)
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

        total_loss /= len(iterator)
        return total_loss[0], acc, rec, pre, acc_box

    def load_bbox(self, file_num):
        with open("../data/imgs/origin_noise_bbox/{:06d}.pickle".format(int(file_num)), "rb") as fin:
            bbox = pickle.load(fin)
        return bbox


    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, res):
        with open("res/{}.json".format(self.model_name), "w") as fout:
            json.dump(res, fout)

    def run(self):
        alpha = [1-self.loss_alpha for _ in range(self.classes + 1)]
        alpha[0] = self.loss_alpha
        loss_func = FocalLoss(gamma=self.loss_gamma, alpha=alpha)
        optimizer = RAdam(self.model.parameters(), lr=self.learning_rate)
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
            if vali_loss < tot_vali_loss:
                tot_vali_loss = vali_loss
                test_loss, acc, rec, pre, acc_box = self.validate(epoch, self.test_loader, loss_func)

                res["best"] = dict(train_loss=float(train_loss), vali_loss=float(vali_loss),
                                test_loss=float(test_loss),
                              accuracy=float(acc), recall=float(rec), precision=float(pre),
                              accuracy_bbox=float(acc_box))
                save_dict = dict(
                    state_dict = self.model.module.state_dict(),
                    epoch=epoch,
                    train_loss=train_loss,
                    vali_loss=vali_loss,
                )
                torch.save(save_dict, os.path.join('weight', self.model_name+".pth"))

            self.save(res)


if __name__ == "__main__":
    fire.Fire(Train)

