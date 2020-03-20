import os
import sys
import tqdm
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.append("../")
from curve import Curve
from folded import Folded

class Pipeline:

    def __init__(self, width=960, height=1280):
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
        return imgs.reshape(-1, self.height, self.width, 3), ys.reshape(-1, self.height // 2, self.width // 2)

    def transform(self, imgs, ys):
        assert imgs.shape[0] == ys.shape[0]
        batch = imgs.shape[0]
        for b in tqdm.tqdm(range(batch)):
            if b % 2 == 0:
                curve_random = random.randint(5, 20)
                curve_mode = "down" if curve_random > 10 else "up"
                tran = Curve(width=self.width, height=self.height, spacing=100,
                          flexure=curve_random/100, direction=curve_mode)
            else:
                folded_up = random.randint(5, 20)
                folded_down = random.randint(5, 20)
                tran = Folded(width=self.width, height=self.height, spacing=100,
                              up_slope=folded_up/100, down_slope=folded_down/100)
            imgs[b] = tran.run(3, imgs[b], ipt_format="opencv", opt_format="opencv")
            y = tran.run(1, cv2.resize(ys[b], (self.width, self.height)).reshape(self.height, self.width, 1),
                         ipt_format="opencv", opt_format="opencv")
            ys[b] = cv2.resize(y, (self.width // 2, self.height // 2))

        #heatmap_img = cv2.applyColorMap(ys[0], cv2.COLORMAP_JET)
        #cv2.imwrite("label_curved.png", heatmap_img)
        #cv2.iimwrite("origin_curved.png", imgs[0])
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
            print(batch, imgs.shape, ys.shape)
            imgs, ys = self.transform(imgs, ys)
            self.save(imgs, ys, batch)


if __name__ == "__main__":
    p = Pipeline()
    p.run()


