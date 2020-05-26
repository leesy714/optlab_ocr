import os
import json
import cv2
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import Data
from model import Model3 as Model
from train import accuracy, recall, precision, accuracy_bbox


device = 'cuda:0'


def load_model(classes, path="weight/Model3.pth"):
    model = Model(classes)
    model.load_state_dict(torch.load(path)["state_dict"])
    return model

def load_test_data(num_data=100):
    data = Data()
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

def test(classes=9):
    model = load_model(classes)
    model.to(device)
    model.eval()

    test_loader = load_test_data()
    if not os.path.exists("res/model"):
        os.makedirs("res/model")


    for crafts, ys, imgs, file_num in test_loader:
        file_num = int(file_num[0])
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(device)
        ys = ys.to(device)
        crafts = crafts.to(device)
        x = torch.cat((imgs, crafts), dim=1)
        pred, _ = model(x)
        pred = pred.permute(0, 2, 3, 1)
        _, argmax = pred.max(dim=3)
        bbox = [load_bbox(b) for b in [file_num]]
        acc_box = accuracy_bbox(ys, argmax, bbox)
        acc = accuracy(ys.view(-1, 1), argmax.view(-1, 1))
        rec = recall(ys.view(-1, 1), argmax.view(-1, 1))
        pre = precision(ys.view(-1, 1), argmax.view(-1, 1))
        print("[file-{:06d}] acc: {:.4}, rec: {:.4}, pre: {:.4}, acc_box: {:.4}".format(
            file_num, float(acc), float(rec), float(pre), float(acc_box)))

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
        cv2.imwrite("./res/model/{:06d}_base.png".format(file_num), img)
        cv2.imwrite("./res/model/{:06d}_pred.png".format(file_num), argmax)
        cv2.imwrite("./res/model/{:06d}_craft.png".format(file_num), craft)

if __name__ == "__main__":
    test()
