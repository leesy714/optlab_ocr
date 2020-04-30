import fire

from base_transform import Transform


class Folded(Transform):

    def __init__(self, up_slope=0.1, down_slope=0.2, **kwargs):
        super().__init__(**kwargs)

        self.up_slope = up_slope
        self.down_slope = down_slope

    def slide(self, h):
        max_num = self.height / self.spacing
        x = abs(1 - 2 * h / max_num)
        if h < max_num / 2:
            return x * self.width * self.up_slope
        else:
            return x * self.width * self.down_slope


if __name__ == "__main__":
    fire.Fire(Folded)
