import os
import sys
import tqdm
import random
import pickle
import numpy as np
from os.path import join as pjoin
from PIL import Image, ImageDraw, ImageFont
import cv2

from curve import Curve
from folded import Folded
from rotation import Rotation
from perspective import Perspective

class Pipeline:

    def __init__(self, width=960, height=1280, res_path="imgs"):
        self.width = width
        self.height = height
        self.res_path = res_path

        self.img_postfix = ".jpg"
        self.y_postfix = ".npy"
        self.bbox_postfix = ".pickle"
        self.annot_postfix = '.xml'

    def load_files(self):
        origin_file_list = os.listdir(pjoin(self.res_path, "origin"))
        label_file_list = os.listdir(pjoin(self.res_path, "origin_label"))
        bbox_file_list = os.listdir(pjoin(self.res_path, "origin_bbox"))
        assert len(origin_file_list) == len(label_file_list) == len(bbox_file_list)
        origin_file_list.sort()
        return origin_file_list

    def load_np(self, idx):
        idx = str(idx)
        img = cv2.imread(pjoin(self.res_path, "origin", idx+self.img_postfix), cv2.IMREAD_COLOR)
        y = np.load(pjoin(self.res_path, "origin_label", idx+self.y_postfix))
        with open(pjoin(self.res_path, "origin_bbox", idx+self.bbox_postfix), 'rb') as fin:
            bbox = pickle.load(fin)
        return img.reshape(self.height, self.width, 3), y.transpose(1, 0), bbox

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
        sigma = np.random.random() * 2.0 + 0.01
        img_blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return img_blur

    def salt_n_pepper(self, img, s_vs_p = 0.5, amount = 0.003):
        #img = img.convert('RGB')
        img_arr = np.array(img)

        row,col,ch = img_arr.shape
        out = np.copy(img_arr)
        # Salt mode
        num_salt = np.ceil(amount * img_arr.size * s_vs_p)

        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img_arr.shape[:2]]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* img_arr.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img_arr.shape[:2]]
        out[coords] = 0
        return out


    def line_pepper(self, img, num_lines=2):
        img_arr = np.array(img)
        row,col,ch = img_arr.shape
        out = np.copy(img_arr)
        coords = np.random.randint(0, col - 1, num_lines)
        for c in coords:
            out[:,c]=1
        return out

    def noise_generate(self, img):
        noise_type = ["BW", "Saturation", "ContrastBrightness", "Blur", "SnP", "linePepper"]
        np.random.shuffle(noise_type)
        shuffle_num = np.random.randint(1, len(noise_type))
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
            elif noise == "SnP":
                img = self.salt_n_pepper(img)
            elif noise == "linePepper":
                img = self.line_pepper(img)
            else:
                raise
        return img

    def annotation(self, img, bb, idx):
        string = ""
        string += "<annotation>\n"
        string += "\t<folder>png_noise</folder>\n"
        string += "\t<filename>{}.png</filename>\n".format(idx)
        string += "\t<size>\n"
        string += "\t\t<width>{}</width>\n".format(img.shape[1])
        string += "\t\t<height>{}</height>\n".format(img.shape[0])
        string += "\t\t<depth>{}</depth>\n".format(img.shape[2])

        string += "\t</size>\n"


        for (xmin,ymin),(xmax,ymax), field in bb:
            string += "\t<object>\n"
            string += "\t\t<name>{}</name>\n".format(field)
            string += "\t\t<pose>Unspecified</pose>\n"
            string += "\t\t<truncated>0</truncated>\n"
            string += "\t\t<difficult>0</difficult>\n"
            string += "\t\t<bndbox>\n"
            string += "\t\t\t<xmin>{}</xmin>\n".format(xmin)
            string += "\t\t\t<ymin>{}</ymin>\n".format(ymin)
            string += "\t\t\t<xmax>{}</xmax>\n".format(xmax)
            string += "\t\t\t<ymax>{}</ymax>\n".format(ymax)
            string += "\t\t</bndbox>\n"
            string += "\t</object>\n"

        string += "</annotation>\n"
        return string

    def bbox_transform(self, bb):
        points = []
        for f, x1, y1, x2, y2, x3,y3, x4, y4 in bb:
            xmin = min(x1,x2,x3,x4)
            ymin = min(y1,y2,y3,y4)
            xmax = max(x1,x2,x3,x4)
            ymax = max(y1,y2,y3,y4)
            points.append(((xmin,ymin),(xmax,ymax),f))
        return points

    def bbox_image(self, img, bb):
        img = img.copy()
        for xy, to_xy, field in bb:
            img = cv2.rectangle(img, xy, to_xy, (0,255,0),2)
        return  img

    def transform(self, img, y, bbox):
        img = self.noise_generate(img.copy())
        mode = random.randint(0, 4)

        if mode == 4:
            tran = Perspective(self.width, self.height)
        elif mode % 2 == 0:
            curve_random = random.randint(5, 20)
            curve_mode = "down" if random.randint(0, 1) > 0 else "up"
            tran = Curve(width=self.width, height=self.height, spacing=40,
                         flexure=curve_random/100, direction=curve_mode,
                         is_horizon=(mode%4==0))
        else:
            folded_up = random.randint(5, 20)
            folded_down = random.randint(5, 20)
            tran = Folded(width=self.width, height=self.height, spacing=40,
                          up_slope=folded_up/100, down_slope=folded_down/100,
                          is_horizon=(mode%4==1))
        rot = Rotation(self.width, self.height)
        img = tran.run(3, img, ipt_format="opencv", opt_format="opencv")
        img = rot.run(3, img, ipt_format="opencv", opt_format="opencv")
        y = cv2.resize(y, (self.width, self.height))
        y = tran.run(1, np.expand_dims(y, 2), ipt_format="opencv", opt_format="opencv")
        y = rot.run(1, np.expand_dims(y, 2), ipt_format="opencv", opt_format="opencv")
        y = np.squeeze(y)
        y = cv2.resize(y, (self.width // 2, self.height // 2))
        y = y.transpose(1, 0).astype(np.uint8)
        points, labels = [], []
        for idx, x1, y1, x2, y2, x3, y3, x4, y4 in bbox:
            points.append((x1, y1))
            points.append((x2, y2))
            points.append((x3, y3))
            points.append((x4, y4))
            labels.append(idx)
        bbox = tran.transform_points(points)
        bbox = rot.transform_points(bbox)
        bbox = [(labels[i], *bbox[4*i], *bbox[4*i+1], *bbox[4*i+2], *bbox[4*i+3]) for i in range(len(bbox)//4)]
        heatmap_img = np.clip(y.transpose(1, 0) * (255 /9), 0 ,255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        cv2.imwrite("label_pers_ori.png", img)
        cv2.imwrite("label_pers.png", heatmap_img)
        return img, y, bbox

    def save(self, img, y, bbox, idx=0):
        idx = str(idx)
        origin_path = pjoin(self.res_path, "origin_noise")
        label_path = pjoin(self.res_path, "origin_noise_label")
        bbox_path = pjoin(self.res_path, "origin_noise_bbox")
        annot_path = pjoin(self.res_path, "origin_noise_annotation")
        bbox_tran = self.bbox_transform(bbox)
        annot = self.annotation(img, bbox_tran, idx)

        for path in [origin_path, label_path, bbox_path, annot_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        cv2.imwrite(pjoin(origin_path, idx+self.img_postfix), img)
        np.save(pjoin(label_path, idx+self.y_postfix),  y)
        with open(pjoin(bbox_path, idx+self.bbox_postfix), 'wb') as fout:
            pickle.dump(bbox, fout)
        with open(pjoin(annot_path, idx+self.annot_postfix),'w') as fout:
            fout.write(annot)

    def run(self):
        for file_name in tqdm.tqdm(self.load_files()):
            idx = file_name.split(".")[0]
            img, y, bbox = self.load_np(idx)
            img, y, bbox = self.transform(img, y, bbox)
            self.save(img, y, bbox, idx)


if __name__ == "__main__":
    p = Pipeline()
    p.run()


