# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import cv2
import numpy as np
import pickle
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

from data import Data
from model import Model3 as Model
from train import accuracy, recall, precision, accuracy_bbox

sys.path.append("../data")
from craft import CRAFT


device = 'cuda:0'

def load_model(classes, path="weight/Model3.pth"):
    model = Model(classes)
    model.load_state_dict(torch.load(path)["state_dict"])
    return model

def load_origin_img(file_path):
    base = cv2.imread(file_path)
    return base

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def craft_inference(net, image):
    x = normalizeMeanVariance(image)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)    # [b, h, w, c] to [b, c, h, w]
    x = Variable(x).cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    y = torch.sum(y, dim=3)
    y = y.squeeze()
    return y

def load_craft_model():
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load("../data/weights/craft_mlt_25k.pth")))
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False
    net.eval()
    return net

def test(classes=9, img_path="test"):
    model = load_model(classes)
    model.to(device)
    model.eval()
    craft_model = load_craft_model()

    test_file_list = os.listdir(img_path)
    for file_num in test_file_list:
        if not file_num.endswith("jpg"):
            continue
        img = load_origin_img(os.path.join(img_path, file_num))
        img = img[:1280*3, :960*3, :]
        H, W, C = img.shape
        _H = H // 6
        _W = W // 6
        _img = np.zeros((_H, _W, C))
        for i in range(3):
            _img[:, :, i] = img[:_H*6, :_W*6, i].reshape(_H, 6, _W, 6).mean(axis=(1, 3))
        img = _img
        img_craft = cv2.resize(img, dsize=(960, 1280), interpolation=cv2.INTER_AREA)
        img_craft = img_craft.reshape(1, 1280, 960, 3)
        craft = craft_inference(craft_model, img_craft)
        img_ipt = cv2.resize(img, dsize=(480, 640), interpolation=cv2.INTER_AREA)
        img_ipt = normalizeMeanVariance(img_ipt)
        img_ipt = img_ipt.reshape(1, 640, 480, 3)
        _img = torch.from_numpy(img_ipt).permute(0, 3, 1, 2)
        _img = _img.to(device)
        craft = craft.to(device)
        craft = craft.view(1, 1, 640, 480)
        x = torch.cat((_img, craft), dim=1)
        pred, _ = model(x)
        pred = pred.permute(0, 2, 3, 1)
        _, argmax = pred.max(dim=3)

        argmax = argmax.cpu().data.numpy()
        craft = craft.squeeze().cpu().data.numpy()
        argmax = argmax[0]

        np.save("./test/res/{}_pred.npy".format(file_num),  argmax)

        craft = np.clip(craft * 255, 0, 255).astype(np.uint8)
        craft = cv2.applyColorMap(craft, cv2.COLORMAP_JET)
        argmax = np.clip(argmax*(255 / classes), 0, 255).astype(np.uint8)
        argmax = cv2.applyColorMap(argmax, cv2.COLORMAP_JET)
        cv2.imwrite("./test/res/{}_craft.png".format(file_num), craft)
        cv2.imwrite("./test/res/{}_pred.png".format(file_num), argmax)
        cv2.imwrite("./test/res/{}_resize.png".format(file_num), img)

if __name__ == "__main__":
    test()
