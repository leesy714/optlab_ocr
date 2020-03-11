import cv2
from PIL import Image
import numpy as np
from os.path import join as pjoin


class Transform:

    def __init__(self, ipt, opt="sample.png", spacing=20,
                 ipt_format="PIL", opt_format="PIL"):
        """
        image with matrix transformation
        :param img: img object
            If img is "opencv", img format should be numpy array
            If img is "PIL", img format should be PIL.Image object
            If img is "file", img format should be string file name.
        :param opt: when opt_format is "file", opt should be file name.
        :param ipt_format: ipt_format is one of ["opencv", "PIL", "file"]
        :param opt_format: opt_format is one of ["opencv", "PIL", "file"]
        :param spacing: length of each square pixel
        """
        if ipt_format == "opencv":
            assert len(opt.shape) == 3
            self.origin = ipt
        elif ipt_format == "PIL":
            self.origin = np.asarray(ipt)
        elif ipt_format == "file":
            self.origin = cv2.imread(pjoin("data", ipt))
        else:
            raise ValueError("ipt_format should be one of ['opencv', 'PIL', 'file']")
        if opt_format not in ["opencv", "PIL", "file"]:
            raise ValueError("opt_format should be one of ['opencv', 'PIL', 'file']")

        self.opt = opt
        self.opt_format = opt_format

        self.width = self.origin.shape[1]
        self.height = self.origin.shape[0]
        print("width: ", self.width, "height: ", self.height)

        self.spacing = spacing

    def slide(self, h):
        raise NotImplemented

    def get_transform(self):
        widths = [[w for h in range(0, self.height, self.spacing)]
                  for w in range(0, self.width, self.spacing)]
        heights = [[h for h in range(0, self.height, self.spacing)]
                   for w in range(0, self.width, self.spacing)]
        return widths, heights

    def save(self, img):
        if self.opt_format == "opencv":
            return img
        elif self.opt_format == "PIL":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        elif self.opt_format == "file":
            cv2.imwrite(pjoin("res", self.opt), img)
            return None

    def run(self):
        output = np.zeros((self.height, self.width, 3), np.uint8)
        print("output", output.shape)
        widths, heights = self.get_transform()
        for w in range(int(self.width / self.spacing)):
            for h in range(int(self.height / self.spacing)):
                width = widths[w][h]
                height = heights[w][h]
                src = np.array([[width, height],
                                [width + self.spacing, height],
                                [width + self.spacing, height + self.spacing],
                                [width, height + self.spacing]], np.float32)
                dst = np.array([[width + self.slide(h), height],
                                [width + self.spacing + self.slide(h), height],
                                [width + self.spacing + self.slide(h+1), +height + self.spacing],
                                [width + self.slide(h+1), height + self.spacing]], np.float32)

                mat = cv2.getPerspectiveTransform(src, dst)
                warp = cv2.warpPerspective(self.origin, mat, (self.width, self.height))
                output[height: height + self.spacing, width:width + self.spacing] = warp[height: height + self.spacing, width:width+self.spacing]
        return self.save(output)

