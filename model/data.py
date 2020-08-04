import os
import sys
import cv2
import csv
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader


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

class LabelData(Dataset):
    CANVAS_SIZE = 1280
    MAG_RATIO = 0.5

    def __init__(self, classes):
        self.base_dir = "../data/imgs/origin_noise/"
        self.ipt_dir = "../data/imgs/origin_craft/"
        self.opt_dir = "../data/imgs/origin_noise_label"
        self.label_dir = "../data/imgs/origin_label"

        assert len(os.listdir(self.ipt_dir)) == len(os.listdir(self.opt_dir)) == len(os.listdir(self.base_dir)),\
            "{}, {}, {}".format(len(os.listdir(self.ipt_dir)), len(os.listdir(self.opt_dir)), len(os.listdir(self.base_dir)))

        self.data_len = len(os.listdir(self.ipt_dir))
        self.classes = classes
        self.mapper = np.eye(classes + 1)

    def split_label(self, label):
        label = label.reshape(-1)
        one_hot_label = self.mapper[label]
        one_hot_label = one_hot_label[:, 1:]
        one_hot_label = one_hot_label.reshape(640, 480, self.classes)
        return one_hot_label

    def __getitem__(self, idx):
        base = cv2.imread(os.path.join(self.base_dir, "{:06d}".format(idx)+".jpg"))
        x = np.load(os.path.join(self.ipt_dir, "{:06d}".format(idx)+".npy"))
        y = np.load(os.path.join(self.opt_dir, "{:06d}".format(idx)+".npy"))
        label = np.load(os.path.join(self.label_dir, "{:06d}".format(idx)+".npy"))

        # noise image, (1280, 960, 3) -> (640, 480, 3) normalized
        base = base.reshape(1, *base.shape)
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio_batch(
            base, self.CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=self.MAG_RATIO)
        base = normalizeMeanVariance(img_resized)
        base = base.squeeze()

        # CRAFR model
        x = x.reshape(1, x.shape[0], x.shape[1])
        y = y.transpose(1, 0)
        label = label.transpose(1, 0)
        label = self.split_label(label)
        label = label.reshape(self.classes, 640, 480)

        return x, y.astype(np.int64), label.astype(np.float32), base, idx

    def __len__(self):
        return self.data_len

class RealData(Dataset):
    CANVAS_SIZE = 1280
    MAG_RATIO = 0.5

    def __init__(self, is_train=True, label=None):
        self.width = 480
        self.height = 640

        folder = "train" if is_train else "test"

        self.boxing = self.get_boxing()
        images, labels, bboxes, crafts, _sizes = dict(), dict(), dict(), dict(), dict()
        base_dir = "../data/AugmentedImage"

        with open(os.path.join(base_dir, "doc_0001_p01.json")) as fin:
            self.label = json.load(fin)

        base_dir = "../data/AugmentedImage/doc_0001"

        for dpi in ["200", "300"]:
            for noise in ["BW", "Color", "Gray"]:
                path = os.path.join(base_dir, folder, dpi, noise, "input")
                files = sorted(os.listdir(path), reverse=True)
                for file_name in files:
                    if "jpg" not in file_name:
                        continue
                    name = file_name.split(".")[0]
                    img = cv2.imread(os.path.join(path, file_name))
                    _sizes[name] = img.shape
                    img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    img = img.reshape(1, *img.shape)
                    img = normalizeMeanVariance(img)
                    img = img.squeeze()
                    images[name] = img

                path = os.path.join(base_dir, folder, dpi, noise, "answer")
                files = sorted(os.listdir(path), reverse=True)
                for file_name in files:
                    name = file_name.split(".")[0]
                    label, bbox = self.boxing_info(os.path.join(path, file_name), _sizes[name])
                    #labels[name] = label.astype(np.int64)
                    bboxes[name] = bbox
                    labels[name] = label

                #path = os.path.join(base_dir, folder, dpi, noise, "label")
                #files = sorted(os.listdir(path), reverse=True)
                #for file_name in files:
                #    name = file_name.split(".")[0]
                #    label = np.load(os.path.join(path, file_name))
                #    print(label.shape)
                #    label = np.expand_dims(label, axis=2)
                #    label = cv2.resize(label, dsize=(self.height, self.width), interpolation=cv2.INTER_NEAREST)
                #    labels[name] = label.astype(np.int64)

                path = os.path.join(base_dir, folder, dpi, noise, "craft")
                files = sorted(os.listdir(path), reverse=True)
                for file_name in files:
                    name = file_name.split(".")[0]
                    craft = np.load(os.path.join(path, file_name))
                    crafts[name] = craft.reshape(1, self.height, self.width)

        self.data_len = len(images)
        print("load images: ", self.data_len)
        self.mapper = {i: name for i, name in enumerate(images.keys())}
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        self.crafts = crafts

    def _parse_field(self, fieldID):
        token = fieldID.split("+")
        front = token[0].split("#")
        front = front[0].split("_")
        if front[-1].isdigit():
            key = "_".join(front[:-1])
        else:
            key = "_".join(front)

        if len(token) > 1:
            key = "{}_{}".format(key, token[1])
        return key

    def get_boxing(self):
        classes = [
            # x1, x2, y1, y2
            (140, 340, 65, 90),
            (40, 125, 147, 161),
            (40, 125, 165, 180),
            (40, 125, 192, 205),
            (40, 125, 216, 230),
            (40, 125, 240, 254),
            (40, 125, 266, 280),
            (40, 125, 345, 363),
        ]
        info = np.zeros((len(classes), self.height, self.width))
        for i, (x1, x2, y1, y2) in enumerate(classes):
            info[i, y1:y2, x1:x2] = 1
        info = info.astype(np.float32)
        return info

    def boxing_info(self, file_name, shape):
        label = np.zeros((self.height, self.width))
        h, w, c = shape
        h_ratio = h / self.height
        w_ratio = w / self.width

        input_file = csv.DictReader(open(file_name))
        bbox = []
        for row in input_file:
            if row.get("FieldID", '1') is not '1':
                boxing = row["Area"]
                fieldID = row["FieldID"]
                if not fieldID:
                    continue
                #if fieldID not in self.label:
                #    self.label[fieldID] = self.label_num
                #    self.label_num = self.label_num + 1

                x1, y1, x2, y2 = boxing.split(";")
                x1, x2 = int(int(x1) / w_ratio), int(int(x2) / w_ratio)
                y1, y2 = int(int(y1) / h_ratio), int(int(y2) / h_ratio)
                label[y1:y2, x1:x2] = self.label[self._parse_field(fieldID)]
                bbox.append((self.label[self._parse_field(fieldID)], x1, y1, x1, y2, x2, y2, x2, y1))
        label = np.expand_dims(label, axis=2)
        label = label.astype(np.int64)
        return label, bbox
        #return bbox

    def get_bbox(self, idx):
        idx = int(idx)
        name = self.mapper[idx]
        return self.bboxes[name]

    def __getitem__(self, idx):
        name = self.mapper[idx]
        return self.images[name], self.crafts[name], self.labels[name], self.boxing, idx

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    real = RealData(is_train=False)
    loader = DataLoader(real, batch_size=2, shuffle=False, num_workers=4)
    for images, crafts, labels, boxing, bboxes in loader:
        print(images.size())
        print(labels.size())
        print(boxing.size())
        label = labels[0].numpy()
        label = np.clip(label * (255 /15), 0 ,255)
        heatmap_img = cv2.applyColorMap(label.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite("origin_label.png", heatmap_img)
        cv2.imwrite("origin.png", images[0].numpy())
        print(bboxes)
        break

