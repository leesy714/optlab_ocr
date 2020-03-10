import fire

from base_transform import Transform


class Folded(Transform):

    def __init__(self, up_slope=0.8, down_slope=0.6, **kwargs):
        super().__init__(**kwargs)

        self.up_slope = up_slope
        self.down_slope = down_slope

    def slide(self, h):
        max_num = self.height / self.spacing
        if h < max_num / 2:
            return (max_num/2 - h) * self.up_slope
        else:
            return (h - max_num/2) * self.down_slope


if __name__ == "__main__":
    fire.Fire(Folded)

