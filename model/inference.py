import os
import json
import cv2
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import Data0001, normalizeMeanVariance
from model import Model0001
from train import accuracy, recall, precision


device = 'cuda:1'

def accuracy_box(y_pred, box):
    label, x1, y1, x2, y2, x3, y3, x4, y4 = box
    x_min, x_max = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
    y_min, y_max = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
    if x_min == x_max or y_min == y_max or y_min >= y_pred.shape[0] or x_min >= y_pred.shape[1]:
        return False
    y_pred_box = np.ravel(y_pred[y_min:y_max, x_min:x_max])
    uniq, counts = np.unique(y_pred_box, return_counts=True)
    uniq, counts = enumerate(list(uniq)), list(counts)
    uniq = sorted(uniq, key=lambda v: counts[v[0]], reverse=True)
    uniq = [u[1] for u in uniq]
    value = uniq[0] if uniq[0] != 0 or len(uniq) <= 1 else uniq[1]
    #acc.append(value != 0)
    return value == label

def IoU(pred, y):
    argmax = np.argmax(pred, axis=2)
    y = np.expand_dims(y, axis=2)
    y_contours, hierachy = cv2.findContours(y, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(y_contours) == 2:
        y_contours = y_contours[0]
    elif len(y_contours) == 3:
        y_contours = y_contours[1]

    argmax[argmax != 0] = 100
    argmax = np.expand_dims(argmax, axis=2).astype(np.float32)
    argmax = cv2.GaussianBlur(argmax, (5, 5), 0)
    argmax[argmax > 30] = 100
    argmax[argmax <= 30] = 0
    argmax = argmax.astype(np.uint8)

    pred_contours, hierachy = cv2.findContours(argmax, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if len(pred_contours) == 2:
        pred_contours = pred_contours[0]
    elif len(pred_contours) == 3:
        pred_contours = pred_contours[1]

    blank1 = np.zeros( (640, 480) )
    blank2 = np.zeros( (640, 480) )
    for contour in y_contours:
        if cv2.contourArea(contour) > 5:
            blank1 = cv2.drawContours(blank1, [contour], -1, 1, -1)
    for contour in pred_contours:
        if cv2.contourArea(contour) > 5:
            blank2 = cv2.drawContours(blank2, [contour], -1, 1, -1)
    intersection = np.sum((blank1 + blank2) == 2)
    union = np.sum((blank1 + blank2) > 0)
    return intersection / (union + 1e-9)


class RealDataTest(Data0001):
    CANVAS_SIZE = 1280
    MAG_RATIO = 0.5

    def __init__(self, is_train=False, label=None):
        self.width = 480
        self.height = 640

        folder = "train" if is_train else "test"

        self.boxing = self.get_boxing()
        images, labels, bboxes, crafts, _sizes = dict(), dict(), dict(), dict(), dict()
        base_dir = "../data/AugmentedImage"

        with open(os.path.join(base_dir, "doc_0001_p01.json")) as fin:
            self.label = json.load(fin)

        path = "../data/cleaned_for_soon_auto_l1"
        path = "../data/cleaned_for_cnn_l1_new"
        files = sorted(os.listdir(path), reverse=True)
        for file_name in files:
            if "jpg" not in file_name or "Business" not in file_name:
                continue
            name = file_name.split(".")[0]
            img = cv2.imread(os.path.join(path, file_name))
            _sizes[name] = img.shape
            img = cv2.resize(img, dsize=(self.width, self.height), interpolation=cv2.INTER_LINEAR)
            img = img.reshape(1, *img.shape)
            img = normalizeMeanVariance(img)
            img = img.squeeze()
            images[name] = img
        base_dir = "../data/AugmentedImage/doc_0001"

        for dpi in ["200", "300"]:
            for noise in ["BW", "Color", "Gray"]:

                path = os.path.join(base_dir, folder, dpi, noise, "answer")
                files = sorted(os.listdir(path), reverse=True)
                for file_name in files:
                    name = file_name.split(".")[0]
                    label, bbox = self.boxing_info(os.path.join(path, file_name), _sizes[name])
                    #labels[name] = label.astype(np.int64)
                    bboxes[name] = bbox
                    labels[name] = label

                path = os.path.join(base_dir, folder, dpi, noise, "craft")
                files = sorted(os.listdir(path), reverse=True)
                for file_name in files:
                    name = file_name.split(".")[0]
                    craft = np.load(os.path.join(path, file_name))
                    crafts[name] = craft.reshape(1, self.height, self.width)

        self.data_len = len(images)
        print("load images: ", self.data_len)
        keys = sorted(list(images.keys()))
        self.mapper = {i: name for i, name in enumerate(keys)}
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        self.crafts = crafts



def load_origin_img(file_num):
    base = cv2.imread(os.path.join("../data/imgs/origin_noise", "{:06d}".format(int(file_num))+".jpg"))
    return base

def draw_polygon(img, pred, bbox):
    for box in bbox:
        label, x1, y1, x2, y2, x3, y3, x4, y4 = box
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        right = accuracy_box(pred, box)
        img = cv2.polylines(img, [pts], True, (0, 255, 0) if right else (0, 0 ,255))
    return img

def test(classes=None):
    data = Data0001(is_train=False)
    data = Subset(data, list(range(100)))

    path="weight/Model0001.pth"
    model = Model0001(anchor=8,
                      pred_classes=24,
                      group_classes=15)
    model.load_state_dict(torch.load(path)["model_state_dict"])
    model.to(device)
    model.eval()

    test_loader = DataLoader(data, batch_size=1, shuffle=False)
    if not os.path.exists("res/model"):
        os.makedirs("res/model")

    #accs, recs, pres, acc_boxs = 0.0, 0.0, 0.0, 0.0
    recs_label, pres_label = [], []
    recs_group, pres_group = [], []
    IoUs_label, IoUs_group = [], []

    for imgs, crafts, ys, groups, boxing, idx in test_loader:
        file_num = int(idx[0])
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = imgs.to(device)
        ys = ys.to(device)
        groups = groups.to(device)
        crafts = crafts.to(device)
        boxing = boxing.to(device)

        pred_label, pred_group = model(imgs, crafts, boxing)
        pred_label = pred_label.permute(0, 2, 3, 1)
        pred_group = pred_group.permute(0, 2, 3, 1)

        _, argmax_label = pred_label.max(dim=3)
        rec_label = recall(ys.view(-1, 1), argmax_label.view(-1, 1))
        pre_label = precision(ys.view(-1, 1), argmax_label.view(-1, 1))
        rec_label, pre_label = float(rec_label), float(pre_label)
        recs_label.append(rec_label)
        pres_label.append(pre_label)
        f1_label = 2*rec_label*pre_label / (rec_label + pre_label + 1e-8)
        IoU_label = IoU(pred_label.squeeze().cpu().data.numpy(), ys.squeeze().cpu().data.numpy())
        IoUs_label.append(IoU_label)

        _, argmax_group = pred_group.max(dim=3)
        rec_group = recall(groups.view(-1, 1), argmax_group.view(-1, 1))
        pre_group = precision(groups.view(-1, 1), argmax_group.view(-1, 1))
        rec_group, pre_group = float(rec_group), float(pre_group)
        recs_group.append(rec_group)
        pres_group.append(pre_group)
        f1_group = 2*rec_group*pre_group / (rec_group + pre_group + 1e-8)
        IoU_group = IoU(pred_group.squeeze().cpu().data.numpy(), groups.squeeze().cpu().data.numpy())
        IoUs_group.append(IoU_group)

        print("[file-{:04d}] label f1-score: {:.4}, group f1-score: {:.4}, IoUs - l {:.4}, g {:.4}".format(
            file_num, float(f1_label), float(f1_group), float(IoU_label), float(IoU_group)))

        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.squeeze().cpu().data.numpy()
        imgs *= np.array([0.229 * 255.0, 0.224*255.0, 0.225*255.0], dtype=np.float32)
        imgs += np.array([0.485 * 255.0, 0.456*255.0, 0.406*255.0], dtype=np.float32)
        imgs = np.clip(imgs, 0, 255).astype(np.uint8)
        craft = crafts.squeeze().cpu().data.numpy()
        ys = ys.squeeze().cpu().data.numpy()
        groups = groups.squeeze().cpu().data.numpy()
        pred_label = pred_label.squeeze().cpu().data.numpy()
        pred_group = pred_group.squeeze().cpu().data.numpy()

        np.save("./res/model/{:04d}_craft.npy".format(file_num), craft)
        np.save("./res/model/{:04d}_true_label.npy".format(file_num),  ys)
        np.save("./res/model/{:04d}_pred_label.npy".format(file_num), pred_label)
        np.save("./res/model/{:04d}_true_group.npy".format(file_num), groups)
        np.save("./res/model/{:04d}_pred_group.npy".format(file_num), pred_group)
        #np.save("./res/model/{:06d}_pred.npy".format(file_num),  argmax)

        argmax_label = argmax_label.cpu().data.numpy()
        argmax_group = argmax_group.cpu().data.numpy()
        argmax_label, argmax_group, img = argmax_label[0], argmax_group[0], imgs
        craft = np.clip(craft * 255, 0, 255).astype(np.uint8)
        craft = cv2.applyColorMap(craft, cv2.COLORMAP_JET)
        argmax_label = np.clip(argmax_label*(255 / 24), 0, 255).astype(np.uint8)
        argmax_label = cv2.applyColorMap(argmax_label, cv2.COLORMAP_JET)
        argmax_group = np.clip(argmax_group*(255 / 15), 0, 255).astype(np.uint8)
        argmax_group = cv2.applyColorMap(argmax_group, cv2.COLORMAP_JET)
        #argmax_img = draw_polygon(argmax_img, argmax, bbox)

        cv2.putText(argmax_label, "F1_SCORE: {:.4}".format(f1_label), (5, 640 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.putText(argmax_group, "F1_SCORE: {:.4}".format(f1_group), (5, 640 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imwrite("./res/model/{:04d}_base.png".format(file_num), img)
        cv2.imwrite("./res/model/{:04d}_pred_label.png".format(file_num), argmax_label)
        cv2.imwrite("./res/model/{:04d}_pred_group.png".format(file_num), argmax_group)
        cv2.imwrite("./res/model/{:04d}_craft.png".format(file_num), craft)

    #data_len = len(test_loader)
    #print("[TOTAL] acc: {:.4}, rec: {:.4}, pre: {:.4}, acc_box: {:.4}".format(
    #    accs / data_len, recs/data_len, pres/data_len, acc_boxs/data_len))

if __name__ == "__main__":
    test()
