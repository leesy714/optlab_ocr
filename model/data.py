import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

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

class Data(Dataset):
    CANVAS_SIZE = 1280
    MAG_RATIO = 0.5

    def __init__(self):
        self.base_dir = "../data/imgs/origin_noise/"
        self.ipt_dir = "../data/imgs/origin_craft/"
        self.opt_dir = "../data/imgs/origin_noise_label"
        assert len(os.listdir(self.ipt_dir)) == len(os.listdir(self.opt_dir)) == len(os.listdir(self.base_dir)),\
            "{}, {}, {}".format(len(os.listdir(self.ipt_dir)), len(os.listdir(self.opt_dir)), len(os.listdir(self.base_dir)))

        self.data_len = len(os.listdir(self.ipt_dir))

    def noise(self, img):
        img = img.astype(np.float32)
        img += np.random.randn(*img.shape) * 60
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        base = cv2.imread(os.path.join(self.base_dir, "{:06d}".format(idx)+".jpg"))
        base = self.noise(base)
        x = np.load(os.path.join(self.ipt_dir, "{:06d}".format(idx)+".npy"))
        y = np.load(os.path.join(self.opt_dir, "{:06d}".format(idx)+".npy"))

        # noise image, (1280, 960, 3) -> (640, 480, 3) normalized
        base = base.reshape(1, *base.shape)
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio_batch(
            base, self.CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=self.MAG_RATIO)
        base = normalizeMeanVariance(img_resized)
        base = base.squeeze()

        # CRAFR model
        x = x.reshape(1, x.shape[0], x.shape[1])
        y = y.transpose(1, 0)
        return x, y.astype(np.int64), base, idx

    def __len__(self):
        return self.data_len

