import fire
import cv2
from PIL import Image
import numpy as np
from os.path import join as pjoin

from base_transform import Transform


class Shadow(Transform):

    def __init__(self, width, height, spacing=20, sources=[(0, 0)], verbose=False):
        """
        :param width: width of the document
        :param height: height of the document
        :param spacing: length of each square pixel
        :param sources: light sources in list format
            e.g. [(0, 0), (0, 1280)]
        """
        super().__init__(width, height, verbose)
        self.spacing = spacing
        assert len(sources) > 0, "at least one light source is required"
        self.sources = sources

    def get_regularizer(self):
        max_norm = 0.0
        for width, height in [(0, 0), (self.width, 0),
                              (0, self.height), (self.width, self.height)]:
            norms = [(width-x)**2 + (height-y)**2 for x, y in self.sources]
            norm = sum(norms)
            max_norm = max(max_norm, norm)
        return 150.0 / max_norm

    def run(self, ipt, ipt_format="file", opt="res.png", opt_format="file"):
        if opt_format not in ["opencv", "PIL", "file"]:
            raise ValueError("opt_format should be one of ['opencv', 'PIL', 'file']")
        self.load_ipt(ipt, ipt_format)

        self.origin = cv2.resize(self.origin, dsize=(self.width, self.height))
        origin = self.origin.astype(np.float32)
        regularizer = self.get_regularizer()
        if self.verbose:
            print("origin", self.origin.shape)
            print("regularizer", regularizer)
        for w in range(0, self.width, self.spacing):
            for h in range(0, self.height, self.spacing):
                widths = slice(w, min(w+self.spacing, self.width))
                heights = slice(h, min(h+self.spacing, self.height))
                norms = [(w-x)**2 + (h-y)**2 for x, y in self.sources]
                norm = sum(norms)
                origin[heights, widths, :] -= int(regularizer * norm)
        origin = origin.clip(min=0, max=255).astype(np.uint8)
        return self.save(origin, opt, opt_format)

if __name__ ==  "__main__":
    fire.Fire(Shadow)
