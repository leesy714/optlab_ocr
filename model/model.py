import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data(Dataset):

    def __init__(self, classes):
        self.classes = classes
        x, y, base = self.load()
        print("data shape", x.shape, y.shape, base.shape)
        self.x = x
        self.y = y
        self.base = base
        self.data_len = x.shape[0]

    def load(self):
        base_dir = "../data/imgs/origin_noise"
        ipt_dir = "../data/imgs/origin_craft"
        opt_dir = "../data/imgs/origin_noise_label"
        file_list = os.listdir(ipt_dir)
        assert len(file_list) == len(os.listdir(opt_dir)) == len(os.listdir(base_dir))
        total_imgs, total_ys, total_bases = [], [], []
        print("start data loading")
        for idx in tqdm(file_list):
            bases = np.fromstring(open(os.path.join(base_dir, idx), "rb").read(), dtype=np.uint8)
            imgs = np.fromstring(open(os.path.join(ipt_dir, idx), "rb").read(), dtype=np.float32)
            ys = np.fromstring(open(os.path.join(opt_dir, idx), "rb").read(), dtype=np.uint8)
            imgs = imgs.reshape(-1, 640, 480, 1)
            total_imgs.append(imgs)
            total_ys.append(ys.reshape(-1, 640, 480))
            total_bases.append(bases.reshape(-1, 1280, 960, 3))
        total_imgs = np.concatenate(total_imgs)
        total_ys = np.concatenate(total_ys)
        total_bases = np.concatenate(total_bases)

        return np.transpose(total_imgs, (0, 3, 1, 2)), total_ys.astype(np.int64), total_bases

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.base[idx]

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


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=(9, 7), padding=(4, 3)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=(9, 7), padding=(4, 3)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Model(nn.Module):

    def __init__(self, classes):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(9, 7), padding=(4, 3)),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        self.conv1 = double_conv(4, 8, 16)


        self.conv_cls = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(16, classes + 1, kernel_size=1),
        )

        init_weights(self.conv.modules())
        init_weights(self.conv1.modules())
        #init_weights(self.conv2.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        y = self.conv(x)
        feature = self.conv1(y)
        y = self.conv_cls(feature)

        return y, feature

class Train:

    def __init__(self, classes, batch_size=4, epochs=10):
        self.classes = classes
        self.epochs =  epochs
        self.learning_rate = 0.005
        self.data = Data(self.classes)

        data_len = len(self.data)
        self.train_num = int(data_len * 0.8)
        self.vali_num = int(data_len * 0.1)
        self.test_num = data_len - self.train_num - self.vali_num
        train, vali, test = random_split(self.data, [self.train_num, self.vali_num, self.test_num])
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
        self.vali_loader = DataLoader(vali, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

        self.model = Model(self.classes).to(device)


    def train(self, epoch, loss_func, optimizer):
        self.model.train()

        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc="({0:^3})".format(epoch))

        for batch, (imgs, ys, _) in pbar:
            imgs = imgs.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()
            pred, _ = self.model(imgs)
            loss = loss_func(pred, ys)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= (self.train_num)
        return total_loss[0]

    def validate(self, epoch, iterator, loss_func):
        self.model.eval()
        total_loss = torch.Tensor([0])

        for batch, (imgs, ys, _) in enumerate(iterator):
            imgs = imgs.to(device)
            ys = ys.to(device)
            pred, _ = self.model(imgs)
            loss = loss_func(pred, ys)
            total_loss += loss.item()
        total_loss /= len(iterator)
        return total_loss[0]

    def test(self):
        self.model.eval()

        for batch, (imgs, ys, bases) in enumerate(self.test_loader):
            imgs = imgs.to(device)
            pred, _ = self.model(imgs)
            pred = pred.permute(0, 2, 3, 1)
            _, argmax = pred.max(dim=3)
            print(argmax.size())
            argmax = argmax.cpu().data.numpy()
            imgs = imgs.squeeze().cpu().data.numpy()

            bases = bases.squeeze().cpu().data.numpy()
            for idx in range(argmax.shape[0]):
                img = argmax[idx]
                origin = np.clip(imgs[idx] * 255, 0, 255).astype(np.uint8)
                img = np.clip(img*(255 / self.classes), 0, 255).astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                origin = cv2.applyColorMap(origin, cv2.COLORMAP_JET)
                cv2.imwrite("./res/{}_base.png".format(idx), bases[idx])
                cv2.imwrite("./res/{}.png".format(idx), img)
                cv2.imwrite("./res/{}_craft.png".format(idx), origin)
            break

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def run(self):
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=0)
        print(self.model)
        print("parameters: ",self.count_parameters())
        tot_vali_loss = np.inf
        for epoch in range(self.epochs):
            train_loss = self.train(epoch, loss_func, optimizer)
            vali_loss = self.validate(epoch, self.vali_loader, loss_func)
            print("train loss: {:.4} vali loss: {:.4}".format(train_loss, vali_loss))
        self.test()

if __name__ == "__main__":
    train = Train(classes=10, epochs=25, batch_size=8)
    train.run()

