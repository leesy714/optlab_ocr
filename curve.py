import fire

from base_transform import Transform


class Curve(Transform):

    def __init__(self, flexure=128, direction="down", **kwargs):
        """
        cureve origin paper to curved paper
        :param flexure: maximum shifted pixel
        :param direction: curve direction
        """
        super().__init__(**kwargs)

        self.flexure = flexure
        self.direction = direction

    def slide(self, h):
        if self.direction == "up":
            return (1 - h / float(self.height / self.spacing)) ** 2 * self.flexure
        elif self.direction == "down":
            return (h / float(self.height / self.spacing)) ** 2 * self.flexure
        else:
            raise KeyError


if __name__ == "__main__":
    fire.Fire(Curve)
