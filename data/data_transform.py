import os
import sys
import tqdm
import random
import numpy as np
from os.path import join as pjoin
from PIL import Image, ImageDraw, ImageFont
import cv2

from curve import Curve
from folded import Folded

class Pipeline:

    def __init__(self, width=960, height=1280, res_path="imgs"):
        self.width = width
        self.height = height

        self.res_path = res_path

    def load_files(self):
        origin_file_list = os.listdir(pjoin(self.res_path, "origin"))
        label_file_list = os.listdir(pjoin(self.res_path, "origin_label"))
        assert len(origin_file_list) == len(label_file_list)
        return origin_file_list

    def load_np(self, idx):
        imgs = np.fromstring(open(pjoin(self.res_path, "origin", idx), "rb").read(), dtype=np.uint8)
        ys = np.fromstring(open(pjoin(self.res_path, "origin_label", idx), "rb").read(), dtype=np.uint8)
        return imgs.reshape(-1, self.height, self.width, 3), ys.reshape(-1, self.height // 2, self.width // 2)

    def bw_global(self, img_gray):
        i = random.randint(0, 2) # global binarization
        thresh = 100 + 28 * i
        ret, img_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
        return img_bw

    def bw_adaptive(self, img_gray, bw_type="Gaussian"):
        block_size = 11

        C = 3 + 2 * random.randint(0, 2)
        if bw_type == "Gaussian":
            img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
        elif bw_type == "Mean":
            img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
        else:
            raise
        img_bw = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB).astype(np.uint8)
        return img_bw

    def saturation(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        orgH, orgS, orgV = cv2.split(img_hsv)

        orgS = orgS.astype("float32")
        factor = 0.25 * (1 + random.randint(0, 6))
        s = np.clip(orgS * factor, 0, 255).astype(np.uint8)

        img_hsv = cv2.merge([orgH, s, orgV])
        img_saturated = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
        return img_saturated

    def contrast_brightness(self, img):
        const_start = 0.25
        const_end = 2.0
        contrastStep = 0.25
        brightnessStep = 25

        contrastRange = np.arange(const_start, const_end + contrastStep, contrastStep)
        c = contrastRange[np.random.randint(len(contrastRange))]

        brightness_start = max(-100, -100 * (c - 0.25))
        brightness_end = min(100, 100 * (1.75 - c))
        brightnessRange = np.arange(brightness_start, brightness_end + brightnessStep, brightnessStep)
        b = brightnessRange[np.random.randint(len(brightnessRange))]
        img = np.clip(img.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
        return img

    def blur(self, img):
        sigma = np.random.randint(2, 4)
        img_blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return img_blur

    def noise_generate(self, img):
        noise_type = ["BW", "Saturation", "ContrastBrightness", "Blur"]
        np.random.shuffle(noise_type)
        shuffle_num = np.random.randint(1, 4)
        for noise in noise_type[:shuffle_num]:
            if noise == "BW":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                bw_type_list = ["Gaussian", "Mean"]
                img = self.bw_adaptive(img, bw_type_list[shuffle_num % 2])
            elif noise == "Saturation":
                img = self.saturation(img)
            elif noise == "ContrastBrightness":
                img = self.contrast_brightness(img)
            elif noise == "Blur":
                img = self.blur(img)
            else:
                raise
        return img

    def transform(self, imgs, ys):
        assert imgs.shape[0] == ys.shape[0]
        batch = imgs.shape[0]
        for b in range(batch):

            imgs[b] = self.noise_generate(imgs[b].copy())

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
        #cv2.imwrite("origin_curved.png", imgs[0])

        return imgs, ys

    def save(self, imgs, ys, idx=0):
        origin_path = pjoin(self.res_path, "origin_noise")
        label_path = pjoin(self.res_path, "origin_noise_label")
        if not os.path.exists(origin_path):
            os.makedirs(origin_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        with open(os.path.join(origin_path, str(idx)), "wb") as fout:
            fout.write(imgs.tostring())
        with open(os.path.join(label_path, str(idx)), "wb") as fout:
            fout.write(ys.tostring())

    def run(self):

        for batch in tqdm.tqdm(self.load_files()):
            imgs, ys = self.load_np(batch)
            imgs, ys = self.transform(imgs, ys)
            self.save(imgs, ys, batch)


if __name__ == "__main__":
    p = Pipeline()
    p.run()


