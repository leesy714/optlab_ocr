import cv2
from PIL import Image
import numpy as np
from os.path import join as pjoin


class Transform:

    def __init__(self, width, height, spacing=40, verbose=False):
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
        self.width = width
        self.height = height
        if verbose:
            print("width: ", self.width, "height: ", self.height)

        self.spacing = spacing
        self.verbose = verbose

    def slide(self, h):
        raise NotImplemented

    def get_transform(self):
        widths = [[w for h in range(0, self.height, self.spacing)]
                  for w in range(0, self.width, self.spacing)]
        heights = [[h for h in range(0, self.height, self.spacing)]
                   for w in range(0, self.width, self.spacing)]
        return widths, heights

    def load_ipt(self, ipt, ipt_format):
        if ipt_format == "opencv":
            assert len(ipt.shape) == 3
            self.origin = ipt
        elif ipt_format == "PIL":
            self.origin = np.asarray(ipt)
        elif ipt_format == "file":
            self.origin = cv2.imread(pjoin("data", ipt))
        else:
            raise ValueError("ipt_format should be one of ['opencv', 'PIL', 'file']")

    def save(self, img, opt, opt_format):
        if opt_format == "opencv":
            return img
        elif opt_format == "PIL":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        elif opt_format == "file":
            cv2.imwrite(pjoin("res", opt), img)
            return None

    def transform_points(self, points):
        """
        :param points: list of points, [(x, y), ... ]
        """
        if not isinstance(points, list):
            points = [points]
        t_points = []
        widths, heights = self.get_transform()
        for point in points:
            w = point[0] // self.spacing
            h = point[1] // self.spacing
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
            point = np.array((*point, 1))
            t_point = np.dot(mat, point)
            t_points.append((int(t_point[0]), int(t_point[1])))
        return t_points

    def run(self, channel, ipt, ipt_format="file", opt="res.png", opt_format="file"):
        if opt_format not in ["opencv", "PIL", "file"]:
            raise ValueError("opt_format should be one of ['opencv', 'PIL', 'file']")
        self.load_ipt(ipt, ipt_format)

        if channel > 1:
            output = np.zeros((self.height, self.width, channel), np.uint8)
        else:
            output = np.zeros((self.height, self.width), np.uint8)
        if self.verbose:
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
        return self.save(output, opt, opt_format)
