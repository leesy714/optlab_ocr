"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image

import cv2
import numpy as np
import json

from craft import CRAFT

from collections import OrderedDict
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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')

args = parser.parse_args()



def resize_aspect_ratio_batch(imgs, square_size, interpolation, mag_ratio=1):
    batch, height, width, channel = imgs.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((batch, target_h32, target_w32, channel), dtype=np.float32)

    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    for b, img in enumerate(imgs):
        proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)
        resized[b, 0:target_h, 0:target_w, :] = proc

    return resized, ratio, size_heatmap

def test_inference(net, image):

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio_batch(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)    # [b, h, w, c] to [b, c, h, w]
    x = Variable(x).cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    y = torch.sum(y, dim=3)
    y = y.squeeze()
    return y.cpu().data.numpy()



def inference(width=1280, height=960, res_path="imgs")
    # load net
    net = CRAFT()

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    output_path = os.path.join(res_path, "origin_craft")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    origin_file_list = os.listdir(os.path.join(res_path, "origin_noise")
    for batch in tqdm(origin_file_list):
        imgs = np.fromstring(open(os.path.join(res_path, "origin_noise", batch), "rb").read(), dtype=np.uint8)
        imgs = imgs.reshape(-1, 1280, 960, 3)
        res = test_inference(net, imgs)

        with open(os.path.join(output_path, str(batch)), "wb") as fout:
            fout.write(res.tostring())

if __name__ == '__main__':
    inference()
