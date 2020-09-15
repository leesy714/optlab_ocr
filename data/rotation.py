import cv2
from PIL import Image
import numpy as np
from os.path import join as pjoin

from base_transform import Transform


class Rotation(Transform):

    def __init__(self, width, height, max_rotate=15, verbose=False):
        super().__init__(width, height, verbose)
        rotate = np.random.randint(max_rotate * 2) - max_rotate
        self.M = cv2.getRotationMatrix2D((width//2, height//2), rotate, 1.0)

    def get_transform(self):
        widths = [[w for h in range(0, self.height, self.spacing)]
                  for w in range(0, self.width, self.spacing)]
        heights = [[h for h in range(0, self.height, self.spacing)]
                   for w in range(0, self.width, self.spacing)]
        return widths, heights

    def transform_points(self, points):
        """
        :param points: list of points, [(x, y), ... ]
        """
        if not isinstance(points, list):
            points = [points]
        t_points = []
        for point in points:
            point = np.array((*point, 1))
            t_point = np.dot(self.M, point)
            t_points.append((int(t_point[0]), int(t_point[1])))
        return t_points

    def run(self, channel, ipt, ipt_format="file", opt="res.png", opt_format="file"):
        if opt_format not in ["opencv", "PIL", "file"]:
            raise ValueError("opt_format should be one of ['opencv', 'PIL', 'file']")
        self.load_ipt(ipt, ipt_format)

        self.origin = cv2.resize(self.origin, dsize=(self.width, self.height))
        if self.verbose:
            print("origin", self.origin.shape)
        output = cv2.warpAffine(self.origin, self.M, (self.width, self.height), flags = cv2.INTER_NEAREST)

        return self.save(output, opt, opt_format)

