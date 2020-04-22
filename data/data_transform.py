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

    def __init__(self, width=960, height=1280, res_path="/home/leesy714/data/ocr/imgs"):
        self.width = width
        self.height = height

        self.res_path = res_path

        self.origin_path = pjoin(self.res_path, "origin_noise")
        self.label_path = pjoin(self.res_path, "origin_noise_label")
        self.png_path = pjoin(self.res_path, "png_noise")
        self.bb_path = pjoin(self.res_path, "annotations")
        self.bb_png_path = pjoin(self.res_path, "bb_noise_png")
        if not os.path.exists(self.origin_path):
            os.makedirs(self.origin_path)
        if not os.path.exists(self.label_path):
            os.makedirs(self.label_path)

        if not os.path.exists(self.png_path):
            os.makedirs(self.png_path)
        if not os.path.exists(self.bb_path):
            os.makedirs(self.bb_path)
        if not os.path.exists(self.bb_png_path):
            os.makedirs(self.bb_png_path)

        self.idx = 0


    def load_files(self):
        origin_file_list = os.listdir(pjoin(self.res_path, "origin"))
        label_file_list = os.listdir(pjoin(self.res_path, "origin_label"))
        assert len(origin_file_list) == len(label_file_list)
        return sorted([int(i) for i in origin_file_list])

    def load_np(self, idx):
        imgs = np.fromstring(open(pjoin(self.res_path, "origin", str(idx)), "rb").read(), dtype=np.uint8)
        ys = np.fromstring(open(pjoin(self.res_path, "origin_label", str(idx)), "rb").read(), dtype=np.uint8)
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
        #img = img.convert('RGB')

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

    def annotation(self, img, bb):
        string = ""
        string += "<annotation>\n"
        string += "\t<folder>png_noise</folder>\n"
        string += "\t<filename>{}.png</filename>\n".format(self.idx)
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


    def bbox_transform(self, tran, bb):
        points = []
        for (xmin, ymin), (xmax, ymax), f in bb:
            points.append((xmin,ymin))
            points.append((xmin,ymax))
            points.append((xmax,ymax))
            points.append((xmax,ymin))


        points = tran.transform_points(points)
        new_bb = []
        for i in range(0,len(points),4):
            x1,y1 = points[i]
            x2,y2 = points[i+1]
            x3,y3 = points[i+2]
            x4,y4 = points[i+3]
            xmin = min(x1,x2,x3,x4)
            ymin = min(y1,y2,y3,y4)
            xmax = max(x1,x2,x3,x4)
            ymax = max(y1,y2,y3,y4)
            f = bb[i // 4][2]
            new_bb.append(((xmin,ymin),(xmax,ymax),f))
        return new_bb



    def bbox_image(self, img, bb):
        img = img.copy()
        for xy, to_xy, field in bb:
            img = cv2.rectangle(img, xy, to_xy, (0,255,0),2)
        return  img





    def transform(self, imgs, ys):
        assert imgs.shape[0] == ys.shape[0]
        batch = imgs.shape[0]
        for b in range(batch):

            imgs[b] = self.noise_generate(imgs[b].copy())
            import pickle
            with open(pjoin(self.res_path,"bb","{:06d}.pkl".format(self.idx)),"rb") as r:
                bb = pickle.load(r)

            if np.random.random()<0.05:
                curve_random = random.randint(5, 20)
                curve_mode = "down" if curve_random > 10 else "up"
                tran = Curve(width=self.width, height=self.height, spacing=100,
                          flexure=curve_random/100, direction=curve_mode)
                imgs[b] = tran.run(3, imgs[b], ipt_format="opencv", opt_format="opencv")
                y = tran.run(1, cv2.resize(ys[b], (self.width, self.height)).reshape(self.height, self.width, 1),
                             ipt_format="opencv", opt_format="opencv")
                ys[b] = cv2.resize(y, (self.width // 2, self.height // 2))
                bb=self.bbox_transform(tran, bb)


            elif np.random.random()<0.05/0.95:
                folded_up = random.randint(5, 20)
                folded_down = random.randint(5, 20)
                tran = Folded(width=self.width, height=self.height, spacing=100,
                              up_slope=folded_up/100, down_slope=folded_down/100)
                imgs[b] = tran.run(3, imgs[b], ipt_format="opencv", opt_format="opencv")
                y = tran.run(1, cv2.resize(ys[b], (self.width, self.height)).reshape(self.height, self.width, 1),
                             ipt_format="opencv", opt_format="opencv")
                ys[b] = cv2.resize(y, (self.width // 2, self.height // 2))
                bb = self.bbox_transform(tran, bb)

            xml = self.annotation(imgs[b], bb)
            cv2.imwrite(pjoin(self.png_path, "{:06d}.jpg").format(self.idx),imgs[b])
            bbox_img = self.bbox_image(imgs[b], bb)
            cv2.imwrite(pjoin(self.bb_png_path, "{:06d}.jpg").format(self.idx),bbox_img)
            with open(pjoin(self.bb_path, "{:06d}.xml").format(self.idx), 'w') as w:
                w.write(xml)

            #with open(pjoin(self.bb_path, "{:06d}.pkl").format(self.idx), 'wb') as w:
            #    pickle.dump(bb, w)


            self.idx+=1


            # 이미지 파일로 확인하기 위한 코드
            #heatmap_img = cv2.applyColorMap(ys[0], cv2.COLORMAP_JET)
            #cv2.imwrite("label_curved.png", heatmap_img)
            #cv2.imwrite("origin_curved_1_{}.png".format(b), imgs[b])

        return imgs, ys

    def save(self, imgs, ys, idx=0):
        with open(os.path.join(self.origin_path, str(idx)), "wb") as fout:
            fout.write(imgs.tostring())
        with open(os.path.join(self.label_path, str(idx)), "wb") as fout:
            fout.write(ys.tostring())

    def run(self):

        for batch in tqdm.tqdm(self.load_files()):
            imgs, ys = self.load_np(batch)
            imgs, ys = self.transform(imgs, ys)
            self.save(imgs, ys, batch)


if __name__ == "__main__":
    p = Pipeline()
    p.run()


