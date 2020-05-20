import os
import json
import fire
import cv2
import numpy as np
import pickle
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging

from craft_inference import copyStateDict, resize_aspect_ratio_batch, normalizeMeanVariance
from craft import CRAFT

from focal_loss import FocalLoss


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'


def accuracy(y_true, y_pred):
    total = y_true.size(0)
    correct = (y_true == y_pred).sum().item()
    return correct / (total + 1e-8)

def recall(y_true, y_pred):
    positive = (y_true > 0)
    total = positive.sum().item()
    correct = (positive[y_true == y_pred]).sum().item()
    return correct / (total + 1e-8)

def precision(y_true, y_pred):
    positive = (y_pred > 0)
    total = positive.sum().item()
    correct = (positive[y_true == y_pred]).sum().item()
    return correct / (total + 1e-8)

def accuracy_bbox(y_true, y_pred, bboxes):
    acc = []
    batch = y_true.size(0)
    for b in range(batch):
        _y_true = y_true[b]
        _y_pred = y_pred[b]
        _bbox = bboxes[b]
        for box in _bbox:
            label, x1, y1, x2, y2, x3, y3, x4, y4 = box
            x_min, x_max = min(x1, x2, x3, x4)//2, max(x1, x2, x3, x4)//2
            y_min, y_max = min(y1, y2, y3, y4)//2, max(y1, y2, y3, y4)//2
            if x_min == x_max or y_min == y_max:
                continue
            y_pred_box = torch.flatten(_y_pred[y_min:y_max, x_min:x_max])
            uniq2, counts = torch.unique(y_pred_box, return_counts=True)
            acc.append(int(uniq2[counts.argmax()]) == label)
            #acc.append(label == uniq2[0])
    return sum(acc) / (len(acc) + 1e-8)

class Data(Dataset):
    CANVAS_SIZE = 1280
    MAG_RATIO = 1.5

    def __init__(self):
        #self.classes = classes
        #x, y, base = self.load()
        #print("data shape", x.shape, y.shape, base.shape)
        #self.x = x
        #self.y = y
        #self.base = base
        self.base_dir = "./imgs/origin_noise/"
        self.ipt_dir = "./imgs/origin_craft/"
        #self.base_dir = '/data/imgs/origin_noise'
        #self.ipt_dir = '/data/imgs/origin_noise'
        #self.ipt_dir = "../data/imgs/origin_craft"
        self.opt_dir = "../data/imgs/origin_noise_label"
        #self.opt_dir = '/data/imgs/origin_noise_label'
        assert len(os.listdir(self.ipt_dir)) == len(os.listdir(self.opt_dir)) == len(os.listdir(self.base_dir))

        self.data_len = len(os.listdir(self.ipt_dir))

    def __getitem__(self, idx):
        base = cv2.imread(os.path.join(self.base_dir, "{:06d}".format(idx)+".jpg"))
        #x = cv2.imread(os.path.join(self.ipt_dir, "{:06d}".format(idx)+".jpg"))
        x = np.load(os.path.join(self.ipt_dir, "{:06d}".format(idx)+".npy"))
        y = np.load(os.path.join(self.opt_dir, "{:06d}".format(idx)+".npy"))

        #base = cv2.imread(os.path.join(self.base_dir, "{:d}".format(idx)+".jpg"))
        #x = cv2.imread(os.path.join(self.ipt_dir, "{:d}".format(idx)+".jpg"))
        #x = np.load(os.path.join(self.ipt_dir, "{:06d}".format(idx)+".npy"))
        #y = np.load(os.path.join(self.opt_dir, "{:d}".format(idx)+".npy"))

        #x = x.reshape(640, 480, 1)
        #x = x.transpose(2, 0, 1)
        y = y.transpose(1, 0)


        #base = base.reshape(1280, 960, 3)

        x = x.reshape(1, x.shape[0], x.shape[1])
        base = base.reshape(1, base.shape[0], base.shape[1], base.shape[2])


        img_resized, target_ratio, size_heatmap = resize_aspect_ratio_batch(
            base, self.CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=self.MAG_RATIO)
        ratio_h = ratio_w = 1/target_ratio

        base = normalizeMeanVariance(img_resized)


        base = base.squeeze()
        return x, y.astype(np.int64), base, idx

    def __len__(self):
        return self.data_len


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class res_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=(3,3), padding=(1,1))

        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), padding=(1,1))

        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)


    def forward(self, x):
        skip_x = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x =  x + self.skip_conv(skip_x)
        return x





class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=(3,3), padding=(1, 1)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Model(nn.Module):
    TRAINED_MODEL = 'weights/craft_mlt_25k.pth'

    def __init__(self, classes, pretrained=False, preprocessed=True):

        assert not(pretrained and preprocessed)
        super(Model, self).__init__()


        if not preprocessed:
            self.craft = CRAFT()

            if pretrained:
                print('Loading weights from checkpoint (' + self.TRAINED_MODEL + ')')
                self.craft.load_state_dict(copyStateDict(torch.load(
                    self.TRAINED_MODEL, map_location='cpu')))
            for p in self.craft.parameters():
                p.requires_grad = False
            self.craft.eval()
        self.preprocessed = preprocessed

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = double_conv(16, 32, 64)
        # self.conv2 = double_conv(64, 128, 256)
        self.conv1 = res_block(16, 32)
        self.conv2 = res_block(32, 64)
        self.conv3 = res_block(64, 128)


        self.conv_cls = nn.Sequential(
            #nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            #nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            res_block(128, 64),
            res_block(64, 32),
            nn.Conv2d(32, classes + 1, kernel_size=1),
        )

        init_weights(self.conv.modules())
        init_weights(self.conv1.modules())
        init_weights(self.conv2.modules())
        init_weights(self.conv3.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x, craft_y=None):
        assert not(self.preprocessed and craft_y is None)

        if not self.preprocessed:
            with torch.no_grad():
                craft_y, craft_feature = self.craft(x)
            craft_y = craft_y.permute(0, 3,1,2)
        pool_x = torch.max_pool2d(x, (2,2))
        x = torch.cat([pool_x, craft_y],dim=1)

        y = self.conv(x)
        feature = self.conv1(y)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        y = self.conv_cls(feature)

        return y, feature

class Train:

    def __init__(self, classes, batch_size=4, loss_gamma=0, loss_alpha=0.5,
                 learning_rate=1e-4, epochs=10):
        self.classes = classes
        self.epochs =  epochs
        self.learning_rate = learning_rate
        self.data = Data()
        self.batch_size = batch_size
        self.loss_gamma = loss_gamma
        self.loss_alpha = loss_alpha

        data_len = len(self.data)
        self.train_num = int(data_len * 0.8)
        self.vali_num = int(data_len * 0.1)
        self.test_num = data_len - self.train_num - self.vali_num
        train, vali, test = random_split(self.data, [self.train_num, self.vali_num, self.test_num])
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
        self.vali_loader = DataLoader(vali, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

        self.model = Model(self.classes)
        if torch.cuda.device_count()>1:
            print("Use ", torch.cuda.device_count(), 'GPUs')
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device)


    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))

        for batch, (crafts, ys, imgs, idx) in pbar:
            imgs = imgs.permute(0,3,1,2)
            imgs = imgs.to(device)
            ys = ys.to(device)
            crafts = crafts.to(device)
            optimizer.zero_grad()
            pred, _ = self.model(imgs, craft_y=crafts)
            loss = loss_func(pred, ys)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= len(self.train_loader)
        return total_loss[0]

    def validate(self, epoch, iterator, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])

        accs, recs, pres, acc_boxes = [] ,[], [], []
        for batch, (crafts, ys, imgs, file_num) in enumerate(iterator):
            imgs = imgs.permute(0,3,1,2)
            imgs = imgs.to(device)
            ys = ys.to(device)
            crafts = crafts.to(device)
            pred, _ = self.model(imgs, craft_y=crafts)
            loss = loss_func(pred, ys)
            total_loss += loss.item()
            pred = pred.permute(0, 2, 3, 1)
            _, argmax = pred.max(dim=3)
            bbox = [self.load_bbox(b) for b in file_num]
            acc_box = accuracy_bbox(ys, argmax, bbox)
            acc = accuracy(ys.view(-1, 1), argmax.view(-1, 1))
            rec = recall(ys.view(-1, 1), argmax.view(-1, 1))
            pre = precision(ys.view(-1, 1), argmax.view(-1, 1))
            accs.append(acc)
            acc_boxes.append(acc_box)
            recs.append(rec)
            pres.append(pre)
        acc = sum(accs) / len(accs)
        acc_box = sum(acc_boxes) / len(acc_boxes)
        rec = sum(recs) / len(recs)
        pre = sum(pres) / len(pres)
        print("acc: {}, recall: {}, precision: {}, acc_bbox: {}".format(acc, rec, pre, acc_box))

        total_loss /= len(iterator)
        return total_loss[0], acc, rec, pre, acc_box

    def load_bbox(self, file_num):
        with open("../data/imgs/origin_noise_bbox/{:06d}.pickle".format(int(file_num)), "rb") as fin:
            bbox = pickle.load(fin)
        return bbox

    def test(self):
        self.model.eval()

        for batch, (crafts, ys, imgs, file_num) in enumerate(self.test_loader):
            imgs = imgs.permute(0,3,1,2)
            imgs = imgs.to(device)
            ys = ys.to(device)
            crafts = crafts.to(device)
            pred, _ = self.model(imgs)
            pred = pred.permute(0, 2, 3, 1)
            _, argmax = pred.max(dim=3)
            argmax = argmax.cpu().data.numpy()
            imgs = imgs.squeeze().cpu().data.numpy()

            bases = bases.squeeze().cpu().data.numpy()
            for idx in range(argmax.shape[0]):
                bbox = self.load_bbox(int(file_num[idx]))
                img = argmax[idx]
                origin = np.clip(imgs[idx] * 255, 0, 255).astype(np.uint8)
                img = np.clip(img*(255 / self.classes), 0, 255).astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                origin = cv2.applyColorMap(origin, cv2.COLORMAP_JET)
                cv2.imwrite("./res/{:06d}_base.png".format(idx), bases[idx])
                cv2.imwrite("./res/{:06d}.png".format(idx), img)
                cv2.imwrite("./res/{:06d}_craft.png".format(idx), origin)
            break

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, res):
        with open("res/{}-{}.json".format(self.loss_gamma, self.loss_alpha), "w") as fout:
            json.dump(res, fout)

    def run(self):
        #loss_func = nn.CrossEntropyLoss()
        alpha = [1-self.loss_alpha for _ in range(self.classes + 1)]
        alpha[0] = self.loss_alpha
        loss_func = FocalLoss(gamma=self.loss_gamma, alpha=alpha)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=0)
        print(self.model)
        print("parameters: ",self.count_parameters())
        res = {}
        tot_vali_loss = np.inf
        for epoch in range(self.epochs):
            #torch.save(self.model.state_dict(), os.path.join('..','ocr_faster_rcnn','data','pretrained_model', "cnn_seg.pth"))
            train_loss = self.train(epoch, loss_func, optimizer)
            vali_loss, acc, rec, pre, acc_box = self.validate(epoch, self.vali_loader, loss_func)
            res[epoch] = dict(train_loss=float(train_loss), vali_loss=float(vali_loss),
                              accuracy=float(acc), recall=float(rec), precision=float(pre),
                              accuracy_bbox=float(acc_box))
            print("train loss: {:.4} vali loss: {:.4}, rec: {:.4}, box_acc: {:.4}".format(train_loss, vali_loss, rec, acc_box))
            self.save(res)
            save_dict = dict(
                state_dict = self.model.module.state_dict(),
                epoch=epoch,
                train_loss=train_loss,
                vali_loss=vali_loss,
            )
            torch.save(save_dict, os.path.join('..','ocr_faster_rcnn','data','pretrained_model', "cnn_seg.pth"))

        #if not os.path.exists("weights"):
        #    os.makedirs("weights")
        self.test()

if __name__ == "__main__":
    train = Train(classes=9, epochs=50, batch_size=8, loss_gamma=0.0, loss_alpha=0.25)
    train.run()

