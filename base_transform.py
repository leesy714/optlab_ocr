import cv2
import numpy as np
from os.path import join as pjoin


class Transform:

    def __init__(self, img_file, res_file="sample.png", spacing=20):
        self.origin = cv2.imread(pjoin("data", img_file))
        self.width = self.origin.shape[1]
        self.height = self.origin.shape[0]
        print("width: ", self.width, "height: ", self.height)

        self.spacing = spacing
        self.res_file = res_file

    def slide(self, h):
        raise NotImplemented

    def get_transform(self):
        widths = [[w for h in range(0, self.height, self.spacing)]
                  for w in range(0, self.width, self.spacing)]
        heights = [[h for h in range(0, self.height, self.spacing)]
                   for w in range(0, self.width, self.spacing)]
        return widths, heights

    def save(self, img):
        cv2.imwrite(pjoin("res", self.res_file), img)

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
        self.save(output)

