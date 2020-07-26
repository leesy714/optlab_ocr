import os
import json
import cv2
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import Data, LabelData
from model import Model5 as Model
from train import accuracy, recall, precision, accuracy_bbox


device = 'cuda:1'


def load_model(classes, path="weight/Model5.pth"):
    model = Model(classes)
    model.load_state_dict(torch.load(path)["model_state_dict"])
    return model

def load_test_data(num_data=100, classes=9):
    data = LabelData(classes)
    data_len = len(data)
    test_num = int(data_len * 0.1)
    test_num = min(test_num, num_data)
    test_data = Subset(data, list(range(data_len - test_num, data_len)))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    return test_loader

def load_bbox(file_num):
    with open("../data/imgs/origin_noise_bbox/{:06d}.pickle".format(int(file_num)), "rb") as fin:
        bbox = pickle.load(fin)
    return bbox

def load_origin_img(file_num):
    base = cv2.imread(os.path.join("../data/imgs/origin_noise", "{:06d}".format(int(file_num))+".jpg"))
    return base

def draw_polygon(img, bbox):
    for box in bbox:
        label, x1, y1, x2, y2, x3, y3, x4, y4 = box
        pts = np.array([[x1/2, y1/2], [x2/2, y2/2], [x3/2, y3/2], [x4/2, y4/2]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.polylines(img, [pts], True, (255, 255, 255))
    return img

def test(classes=9):
    model = load_model(classes)
    model.to(device)
    model.eval()

    test_loader = load_test_data(100, classes)
    if not os.path.exists("res/model"):
        os.makedirs("res/model")

    accs, recs, pres, acc_boxs = 0.0, 0.0, 0.0, 0.0
    for crafts, ys, labels, imgs, file_num in test_loader:
        file_num = int(file_num[0])
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(device)
        ys = ys.to(device)
        crafts = crafts.to(device)
        labels = labels.to(device)
        #x = torch.cat((imgs, crafts), dim=1)
        #pred, _ = model(x)
        pred, _ = model(imgs, crafts, labels)
        pred = pred.permute(0, 2, 3, 1)
        _, argmax = pred.max(dim=3)
        bbox = [load_bbox(b) for b in [file_num]]
        acc_box = accuracy_bbox(ys, argmax, bbox)
        acc = accuracy(ys.view(-1, 1), argmax.view(-1, 1))
        rec = recall(ys.view(-1, 1), argmax.view(-1, 1))
        pre = precision(ys.view(-1, 1), argmax.view(-1, 1))
        accs += float(acc)
        recs += float(rec)
        pres += float(pre)
        acc_boxs += float(acc_box)
        print("[file-{:06d}] acc: {:.4}, rec: {:.4}, pre: {:.4}, acc_box: {:.4}".format(
            file_num, float(acc), float(rec), float(pre), float(acc_box)))
        #continue
        argmax = argmax.cpu().data.numpy()
        imgs = imgs.squeeze().cpu().data.numpy()
        craft = crafts.squeeze().cpu().data.numpy()
        ys = ys.squeeze().cpu().data.numpy()
        bbox, argmax, img = bbox[0], argmax[0], imgs[0]

        np.save("./res/model/{:06d}_true.npy".format(file_num),  ys)
        np.save("./res/model/{:06d}_pred.npy".format(file_num),  argmax)
        img = load_origin_img(file_num)

        craft = np.clip(craft * 255, 0, 255).astype(np.uint8)
        craft = cv2.applyColorMap(craft, cv2.COLORMAP_JET)
        argmax = np.clip(argmax*(255 / classes), 0, 255).astype(np.uint8)
        argmax = cv2.applyColorMap(argmax, cv2.COLORMAP_JET)
        argmax = draw_polygon(argmax, bbox)
        cv2.imwrite("./res/model/{:06d}_base.png".format(file_num), img)
        cv2.imwrite("./res/model/{:06d}_pred.png".format(file_num), argmax)
        cv2.imwrite("./res/model/{:06d}_craft.png".format(file_num), craft)

    data_len = len(test_loader)
    print("[TOTAL] acc: {:.4}, rec: {:.4}, pre: {:.4}, acc_box: {:.4}".format(
        accs / data_len, recs/data_len, pres/data_len, acc_boxs/data_len))

if __name__ == "__main__":
    test()
