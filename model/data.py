import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset

sys.path.append("../data")
from craft_inference import copyStateDict, resize_aspect_ratio_batch, normalizeMeanVariance


class Data(Dataset):
    CANVAS_SIZE = 1280
    MAG_RATIO = 0.5

    def __init__(self):
        self.base_dir = "../data/imgs/origin_noise/"
        self.ipt_dir = "../data/imgs/origin_craft/"
        self.opt_dir = "../data/imgs/origin_noise_label"
        assert len(os.listdir(self.ipt_dir)) == len(os.listdir(self.opt_dir)) == len(os.listdir(self.base_dir))

        self.data_len = len(os.listdir(self.ipt_dir))

    def __getitem__(self, idx):
        base = cv2.imread(os.path.join(self.base_dir, "{:06d}".format(idx)+".jpg"))
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

