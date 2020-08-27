import os
import sys
import cv2
import csv
import json
import numpy as np
import collections
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


class Document:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.label = None

    def get_label_info(self, path):
        pass

    def get_image(self, path):
        img = cv2.imread(path)
        origin_shape = img.shape
        img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img = img.reshape(1, *img.shape)
        img = normalizeMeanVariance(img)
        img = img.squeeze()
        return img, origin_shape

    def get_box(self, path, shape):
        label = np.zeros((self.height, self.width))
        h, w, c = shape
        h_ratio = h / self.height
        w_ratio = w / self.width

        input_file = csv.DictReader(open(path))
        bbox = []
        for row in input_file:
            #if row.get("FieldID", '1') is not '1':
            if row.get("Field Name", '') is not '':
                boxing = row["Area"]
                fieldID = row["Field Name"]
                if not fieldID:
                    continue

                x1, y1, x2, y2 = boxing.split(";")
                x1, x2 = int(int(x1) / w_ratio), int(int(x2) / w_ratio)
                y1, y2 = int(int(y1) / h_ratio), int(int(y2) / h_ratio)
                bbox.append((self.label[fieldID], x1, y1, x1, y2, x2, y2, x2, y1))
        return bbox

    def get_label(self, path):
        label = np.load(path)
        label = cv2.resize(label, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        label = np.expand_dims(label, axis=2)
        return label.astype(np.int64)

    def get_group(self, path):
        group = np.load(path)
        group = cv2.resize(group, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        group = np.expand_dims(group, axis=2)
        return group.astype(np.int64)

    def get_craft(self, path):
        craft = np.load(path)
        craft = craft.reshape(1, self.height, self.width)
        return craft

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

    def load(self, base_dir, is_train):
        folder = "train" if is_train else "test"
        images, labels, bboxes, crafts, groups, _sizes = dict(), dict(), dict(), dict(), dict(), dict()
        for dpi in ["200", "300"]:
        #for dpi in ["200"]:
            for noise in ["BW", "Color", "Gray"]:
            #for noise in ["BW"]:
                path = os.path.join(base_dir, folder, dpi, noise, "input")
                for file_name in os.listdir(path):
                    if "jpg" not in file_name:
                        continue
                    name = file_name.split(".")[0]
                    images[name], _sizes[name] = self.get_image(os.path.join(path, file_name))

                path = os.path.join(base_dir, folder, dpi, noise, "answer")
                for file_name in os.listdir(path):
                    name = file_name.split(".")[0]
                    #bboxes[name] = self.get_box(os.path.join(path, file_name), _sizes[name])
                    bboxes[name] = []

                path = os.path.join(base_dir, folder, dpi, noise, "label")
                for file_name in os.listdir(path):
                    name = file_name.split(".")[0]
                    labels[name] = self.get_label(os.path.join(path, file_name))

                path = os.path.join(base_dir, folder, dpi, noise, "group")
                for file_name in os.listdir(path):
                    name = file_name.split(".")[0]
                    groups[name] = self.get_group(os.path.join(path, file_name))

                path = os.path.join(base_dir, folder, dpi, noise, "craft")
                for file_name in os.listdir(path):
                    name = file_name.split(".")[0]
                    crafts[name] = self.get_craft(os.path.join(path, file_name))
        return images, labels, bboxes, crafts, groups

class Data0001(Dataset, Document):

    def __init__(self, is_train=True, label=None, dataset="doc_0001"):
        self.width = 480
        self.height = 640
        self.label = self.get_label_info()

        self.boxing = self.get_boxing()
        base_dir = "../data/AugmentedImage/doc_0001"
        self.images, self.labels, self.bboxes, self.crafts, self.groups = self.load(base_dir, is_train)

        self.data_len = len(self.images)
        print("load images: ", self.data_len)
        keys = sorted(list(self.images.keys()))
        self.mapper = {i: name for i, name in enumerate(keys)}

    def get_label_info(self):
        with open(os.path.join("../data/AugmentedImage", "doc_0001_p01.json"), encoding="utf-8") as fin:
            label = json.load(fin)
        final_label = {}
        label_mapper = {}
        print("label", label)
        max_id = max([v for v in label.values() if v < 100])
        print("max_id", max_id)
        keys = sorted(list(label.keys()))
        for key in keys:
            value = label[key]
            if value >= 100:
                max_id += 1
                final_label[key] = max_id
                label_mapper[value] = max_id
            else:
                final_label[key] = value
        print(label_mapper)
        self.label_mapper = label_mapper
        return final_label

    def get_label(self, path):
        label = np.load(path)
        label = cv2.resize(label, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        _max = 0
        for key, value in self.label_mapper.items():
            label[label==key] = value
            _max = max(_max, key)
        label[label>_max] = 0
        label = np.expand_dims(label, axis=2)
        #print("label", np.max(label))
        return label.astype(np.int64)

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

    def get_bbox(self, idx):
        idx = int(idx)
        name = self.mapper[idx]
        return self.bboxes[name]

    def __getitem__(self, idx):
        name = self.mapper[idx]
        return self.images[name], self.crafts[name], self.labels[name], self.groups[name], self.boxing, idx

    def __len__(self):
        return self.data_len


class Data0002(Dataset, Document):

    def __init__(self, is_train=True, label=None):
        self.width = 480
        self.height = 640
        self.label = self.get_label_info()

        self.boxing = self.get_boxing()
        base_dir = "../data/AugmentedImage/doc_0002"
        self.images, self.labels, self.bboxes, self.crafts, self.groups = self.load(base_dir, is_train)

        self.data_len = len(self.images)
        print("load images: ", self.data_len)
        keys = sorted(list(self.images.keys()))
        self.mapper = {i: name for i, name in enumerate(keys)}

    def get_label_info(self, path):
        with open(os.path.join("../data/AugmentedImage", "doc_0002_p01.json")) as fin:
            label = json.load(fin)
        return label

    def get_boxing(self):
        classes = [
            # x1, x2, y1, y2
            (70, 195, 45, 65),
            (37, 110, 114, 124),
            (270, 320, 110, 128),
            (32, 60, 165, 174),
            (32, 60, 215, 224),
            (70, 90, 215, 224),
            (105, 150, 215, 224),
            (260, 370, 215, 224),
            (380, 440, 215, 224),
        ]
        info = np.zeros((len(classes), self.height, self.width))
        for i, (x1, x2, y1, y2) in enumerate(classes):
            info[i, y1:y2, x1:x2] = 1
        info = info.astype(np.float32)
        return info

    def get_bbox(self, idx):
        idx = int(idx)
        name = self.mapper[idx]
        return self.bboxes[name]

    def __getitem__(self, idx):
        name = self.mapper[idx]
        return self.images[name], self.crafts[name], self.labels[name], self.groups[name], self.boxing, idx

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    real = Data0001(is_train=True)
    loader = DataLoader(real, batch_size=2, shuffle=False, num_workers=4)
    for images, crafts, labels, groups, boxing, idx in loader:
        print(images.size())
        print(labels.size())
        print(boxing.size())
        #label = labels[0].numpy()
        #label = np.clip(label * (255 /15), 0 ,255)
        #heatmap_img = cv2.applyColorMap(label.astype(np.uint8), cv2.COLORMAP_JET)
        #cv2.imwrite("origin_label.png", heatmap_img)
        #cv2.imwrite("origin.png", images[0].numpy())
        print(bboxes)
        break

