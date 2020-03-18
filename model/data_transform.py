import os
import sys
import tqdm
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.append("../")
from curve import Curve


class Pipeline:

    def __init__(self, width=800, height=1200):
        self.width = width
        self.height = height

    def load_files(self):
        origin_file_list = os.listdir("/data/origin")
        label_file_list = os.listdir("/data/origin_label")
        assert len(origin_file_list) == len(label_file_list)
        return origin_file_list

    def load_np(self, idx):
        imgs = np.fromstring(open(os.path.join("/data/origin", idx), "rb").read(), dtype=np.uint8)
        ys = np.fromstring(open(os.path.join("/data/origin_label", idx), "rb").read(), dtype=np.uint8)
        return imgs.reshape(-1, self.height, self.width, 3), ys.reshape(-1, self.height, self.width)

    def transform(self, imgs, ys):
        c = Curve(width=self.width, height=self.height, spacing=100)
        assert imgs.shape[0] == ys.shape[0]
        batch = imgs.shape[0]
        for b in tqdm.tqdm(range(batch)):
            imgs[b] = c.run(3, imgs[b], ipt_format="opencv", opt_format="opencv")
            y = c.run(1, ys[b].reshape(self.height, self.width, 1),
                      ipt_format="opencv", opt_format="opencv")
            ys[b] = y.reshape(self.height, self.width)

        heatmap_img = cv2.applyColorMap(ys[0], cv2.COLORMAP_JET)
        cv2.imwrite("label_curved.png", heatmap_img)
        cv2.imwrite("origin_curved.png", imgs[0])
        return imgs, ys

    def save(self, imgs, ys, idx=0):
        origin_path = "/data/origin_noise"
        label_path = "/data/origin_noise_label"
        if not os.path.exists(origin_path):
            os.makedirs(origin_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        with open(os.path.join(origin_path, str(idx)), "wb") as fout:
            fout.write(imgs.tostring())
        with open(os.path.join(label_path, str(idx)), "wb") as fout:
            fout.write(ys.tostring())

    def run(self):
        for batch in self.load_files():
            imgs, ys = self.load_np(batch)
            print(imgs.shape, ys.shape)
            imgs, ys = self.transform(imgs, ys)
            self.save(imgs, ys, batch)


if __name__ == "__main__":
    p = Pipeline()
    p.run()


