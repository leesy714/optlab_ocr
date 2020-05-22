#TODO: load weight and inference file

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
