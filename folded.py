import cv2
import numpy as np
from os.path import join as pjoin


class Folded:

    def __init__(self, img_file="sample.png", spacing=8):
        self.origin = cv2.imread(pjoin("data", img_file))
        self.width = self.origin.shape[0]
        self.height = self.origin.shape[1]
        self.spacing = spacing

        self.up_slope = 0.8
        self.down_slope = 0.6

    def slide(self, h):
        max_num = self.height / self.spacing
        if h < max_num / 2:
            return (max_num/2 - h) * self.up_slope
        else:
            return (h - max_num/2) * self.down_slope

    def get_transform(self):
        widths = [[w for w in range(0, self.width, self.spacing)]
                  for _ in range(0, self.height, self.spacing)]
        heights = [[h for h in range(0, self.height, self.spacing)]
                   for _ in range(0, self.width, self.spacing)]
        return widths, heights

    def save(self, img):
        cv2.imwrite(pjoin("res", "folded.png"), img)

    def run(self):
        output = np.zeros((self.width, self.height, 3), np.uint8)
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
                output[width: width + self.spacing] = warp[width: width + self.spacing]
                output[height: height + self.spacing] = warp[height: height + self.spacing]
        self.save(output)


if __name__ == "__main__":
    folded = Folded()
    folded.run()
