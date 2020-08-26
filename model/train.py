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

from data import Data0001
from model import Model0001
from focal_loss import FocalLoss
from optimizer import RAdam

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
#device = 'cpu'

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
            #x_min, x_max = min(x1, x2, x3, x4)//2, max(x1, x2, x3, x4)//2
            #y_min, y_max = min(y1, y2, y3, y4)//2, max(y1, y2, y3, y4)//2
            x_min, x_max = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
            y_min, y_max = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
            if x_min == x_max or y_min == y_max or y_min >= _y_pred.size(0) or x_min >= _y_pred.size(1):
                continue
            y_pred_box = torch.flatten(_y_pred[y_min:y_max, x_min:x_max])
            uniq, counts = torch.unique(y_pred_box, return_counts=True)
            uniq, counts = enumerate(list(uniq.cpu().numpy())), list(counts.cpu().numpy())
            uniq = sorted(uniq, key=lambda v: counts[v[0]], reverse=True)
            uniq = [u[1] for u in uniq]
            value = uniq[0] if uniq[0] != 0 or len(uniq) <= 1 else uniq[1]
            acc.append(value == label)
            #acc.append(value != 0)
    return sum(acc) / (len(acc) + 1e-8)


class Train:

    def __init__(self, batch_size=1, loss_gamma=0.1, loss_alpha=0.4,
                 learning_rate=1e-4, epochs=10, load_model=False):
        """
        :param classes: number of classes except background
        :param batch_size: training dataset batch size
        :param loss_gamma: gamma value in focal loss
        :param loss_alpha: alpha value in focal loss
        :param learning_rate: Optimizer learning_rate
        :param epochs: training_epochs
        :param load_model: model pretrain model to restart model training
        """

        self.classes_pred = 24
        self.classes_group = 15

        self.epochs =  epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_gamma = loss_gamma
        self.loss_alpha = loss_alpha

        self.train_data = Data0001(is_train=True)
        self.vali_data = Data0001(is_train=False)
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.vali_loader = DataLoader(self.vali_data, batch_size=batch_size, shuffle=False, num_workers=4)

        #self.classes = len(self.train_data.label)

        self.model = Model0001(anchor=8, pred_classes=self.classes_pred,
                               group_classes=self.classes_group)
        self.model_name = self.model.__class__.__name__

        if torch.cuda.device_count()>1:
            print("Use ", torch.cuda.device_count(), 'GPUs')
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device)

        self.optimizer = RAdam(self.model.parameters(), lr=self.learning_rate)
        self.save_path = os.path.join('weight', self.model_name+".pth")
        self.start_epoch = 0

        if load_model:
            checkpoint = torch.load(self.save_path)
            self.start_epoch = checkpoint["epoch"]
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        alpha = [1-self.loss_alpha for _ in range(self.classes_pred + 1)]
        alpha[0] = self.loss_alpha
        self.loss_pred = FocalLoss(gamma=self.loss_gamma, alpha=alpha)

        alpha = [1-self.loss_alpha for _ in range(self.classes_group + 1)]
        alpha[0] = self.loss_alpha
        self.loss_group = FocalLoss(gamma=self.loss_gamma, alpha=alpha)

    def train(self, epoch):
        self.model.train()

        total_loss = 0.0
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))

        for batch, (imgs, crafts, ys, groups, boxing, _) in pbar:
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.to(device)
            crafts = crafts.to(device)
            boxing = boxing.to(device)
            groups = groups.to(device)
            ys = ys.to(device)
            pred_label, pred_group = self.model(imgs, crafts, boxing)
            loss_label = self.loss_pred(pred_label, ys)

            loss_group = self.loss_group(pred_group, groups)
            loss = loss_label + loss_group
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            batch_loss = loss_label.item() + loss_group.item()
            total_loss += batch_loss
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= len(self.train_loader)
        return total_loss

    def validate(self, epoch, iterator):
        self.model.eval()
        total_loss = 0.0

        recs_label, pres_label = [], []
        recs_group, pres_group = [], []
        for batch, (imgs, crafts, ys, groups, boxing, idx) in enumerate(iterator):
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.to(device)
            crafts = crafts.to(device)
            groups = groups.to(device)
            ys = ys.to(device)
            pred_label, pred_group = self.model(imgs, crafts, boxing)
            loss = self.loss_pred(pred_label, ys) + self.loss_group(pred_group, groups)
            total_loss += loss.item()
            pred_label = pred_label.permute(0, 2, 3, 1)
            pred_group = pred_group.permute(0, 2, 3, 1)

            _, argmax = pred_label.max(dim=3)
            rec = recall(ys.view(-1, 1), argmax.view(-1, 1))
            pre = precision(ys.view(-1, 1), argmax.view(-1, 1))
            recs_label.append(rec)
            pres_label.append(pre)

            _, argmax = pred_group.max(dim=3)
            rec = recall(groups.view(-1, 1), argmax.view(-1, 1))
            pre = precision(groups.view(-1, 1), argmax.view(-1, 1))
            recs_group.append(rec)
            pres_group.append(pre)
        rec_label = sum(recs_label) / len(recs_label)
        pre_label = sum(pres_label) / len(pres_label)
        rec_group = sum(recs_group) / len(recs_group)
        pre_group = sum(pres_group) / len(pres_group)
        f1_label = 2*rec_label*pre_label/(rec_label+pre_label)
        f1_group = 2*rec_group*pre_group/(rec_group+pre_group)

        total_loss /= len(iterator)
        return total_loss, f1_label, f1_group
    
    def test(self):
        self.model.eval()

        count = 0
        for batch, (imgs, crafts, ys, boxing, idx) in enumerate(self.vali_loader):
            imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.to(device)
            crafts = crafts.to(device)
            ys = ys.to(device)
            preds, _ = self.model(imgs, crafts, boxing)
            preds = preds.permute(0, 2, 3, 1)
            _, argmaxs = preds.max(dim=3)
            imgs = imgs.permute(0, 2, 3, 1)
            crafts = crafts.squeeze()
            for img, craft, y, argmax in zip(imgs, crafts, ys, argmaxs):
                img = img.cpu().data.numpy()
                img *= np.array([0.229 * 255.0, 0.224*255.0, 0.225*255.0], dtype=np.float32)
                img += np.array([0.485 * 255.0, 0.456*255.0, 0.406*255.0], dtype=np.float32)
                img = np.clip(img, 0, 255).astype(np.uint8)
                craft = craft.cpu().data.numpy()
                argmax = argmax.cpu().data.numpy()
                craft = np.clip(craft * 255, 0, 255).astype(np.uint8)
                craft = cv2.applyColorMap(craft, cv2.COLORMAP_JET)
                argmax = np.clip(argmax*(255 / len(self.train_data.label)), 0, 255).astype(np.uint8)
                argmax = cv2.applyColorMap(argmax, cv2.COLORMAP_JET)
                cv2.imwrite("./res_temp/{:04d}_base.png".format(count), img)
                cv2.imwrite("./res_temp/{:04d}_pred.png".format(count), argmax)
                cv2.imwrite("./res_temp/{:04d}_craft.png".format(count), craft)
                count = count + 1
            if count >= 100:
                break

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, res):
        with open("res/{}.json".format(self.model_name), "w") as fout:
            json.dump(res, fout)

    def run(self):
        print(self.model)
        print("parameters: ",self.count_parameters())
        res = {}
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train(epoch)
            vali_loss, f1_label, f1_group = self.validate(epoch, self.vali_loader)
            res[epoch] = dict(train_loss=float(train_loss), vali_loss=float(vali_loss),
                              f1_label=float(f1_label), f1_group=float(f1_group))
            print("train loss: {:.4} vali loss: {:.4}, f1_group: {:.4}, f1_label: {:.4}".format(train_loss, vali_loss, f1_group, f1_label))
            save_dict = dict(
                model_state_dict = self.model.module.state_dict(),
                optimizer_state_dict = self.optimizer.state_dict(),
                epoch=epoch,
                train_loss=train_loss,
                vali_loss=vali_loss,
            )
            torch.save(save_dict, self.save_path)
            self.save(res)
        #self.test()

if __name__ == "__main__":
    fire.Fire({
        "train": Train,
    })

