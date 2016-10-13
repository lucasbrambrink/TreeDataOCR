from PIL import Image, ImageDraw


class Stat(object):

    def __init__(self, data):
        self.data = data

    @property
    def mean(self):
        return sum(self.data) / len(self.data)

    @property
    def sqr_dev(self):
        return map(lambda x: x ** 2, self.deviations)

    @property
    def deviations(self):
        mean = self.mean
        return [d - mean for d in self.data]

    @property
    def var(self):
        return sum(self.sqr_dev) / len(self.sqr_dev)

    @property
    def std(self):
        return self.var ** 0.5


class Steps(object):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (1, -1)
    DOWN_LEFT = (-1, 1)
    DOWN_RIGHT = (1, 1)
    CENTER = (0, 0)
    ALL = (UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT)
    MAIN_FOUR = (UP, DOWN, LEFT, RIGHT)

    QUADRANTS = (
        (UP, RIGHT, UP_RIGHT),
        (DOWN, RIGHT, DOWN_RIGHT),
        (DOWN, LEFT, DOWN_LEFT),
        (UP, LEFT, UP_LEFT),
    )

    EDGES = [(7, 0, 1),
             (1, 2, 3),
             (3, 4, 5),
             (5, 6, 7)] # index, step

    VERT = (
        (7, 0, 1),
        (6, -1, 2),
        (5, 4, 3),
    )
    HORIZ = (
        (7, 6, 5),
        (0, -1, 4),
        (1, 2, 3),
    )

    ORIENTATIONS = (
        (QUADRANTS[0], QUADRANTS[2]),
        (QUADRANTS[1], QUADRANTS[3]),
        (QUADRANTS[2], QUADRANTS[0]),
        (QUADRANTS[3], QUADRANTS[1])
    )

    ACCEPTABLE_POS = (
        (UP, UP_LEFT, LEFT), #(UP, UP_RIGHT),
        (RIGHT, UP_RIGHT, UP),# (RIGHT, DOWN_RIGHT),
        (DOWN, DOWN_RIGHT, RIGHT),# (DOWN, DOWN_LEFT),
        (LEFT, DOWN, DOWN_LEFT),# (LEFT, UP_LEFT),
        # (UP, LEFT),
        # (UP, RIGHT),
        # (DOWN, LEFT),
        # (DOWN, RIGHT)
    )

    BORDERS = (
        (UP, (UP_LEFT, UP_RIGHT)),
        (LEFT, (UP_LEFT, DOWN_LEFT)),
        (RIGHT, (UP_RIGHT, DOWN_RIGHT)),
        (DOWN, (DOWN_LEFT, DOWN_RIGHT)),
    )

    @staticmethod
    def step(pixel, step_type):
        return (pixel[0] + step_type[0],
                pixel[1] + step_type[1])

    @classmethod
    def all_step(cls, pixel):
        return [cls.step(pixel, step)
                for step in Steps.ALL]

    @classmethod
    def step_many(cls, pixel, steps=2):
        return [cls.step(pixel, (x, y,))
                for x in range(-steps, steps + 1)
                for y in range(-steps, steps + 1)
                if (x, y) != (0, 0)]

    @classmethod
    def step_all(cls, pixel, polygon):
        neighbors = []
        for step in cls.ALL:
            test = cls.step(pixel, step)
            if test in polygon:
                neighbors.append(test)
        return neighbors

    @classmethod
    def rows(cls, surrounding, center):
        return (cls.row(surrounding, cls.VERT, center),
                cls.row(surrounding, cls.HORIZ, center))

    @staticmethod
    def row(surrounding, mapping, center):
        return [(surrounding[i] if i != -1 else center
                 for i in row) for row in mapping]


class Pixel(object):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    LOWER_BOUND_FOR_WHITENESS = 240
    UPPER_BOUND_FOR_BLACK = 230
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)


    @classmethod
    def is_mostly_black(cls, pixel_values):
        for p in pixel_values:
            if p < cls.UPPER_BOUND_FOR_BLACK:
                return True
        return False

    @classmethod
    def is_mostly_white(cls, pixel_values):
        for p in pixel_values:
            if p > cls.LOWER_BOUND_FOR_WHITENESS:
                return True
        return False

    @staticmethod
    def slope(pixel1, pixel2):
        if (pixel2[0] - pixel1[0]) == 0:
            return float(100)
        return float(pixel2[1] - pixel1[1]) / float(pixel2[0] - pixel1[0])

    @classmethod
    def detailed_slope(cls, p1, p2):
        if (p1[0] - p2[0]) == 0:
            o = 1 if p1[1] < p2[1] else -1
            return float(100), o
        if (p1[1] - p2[1]) == 0:
            o = 1 if p1[0] < p2[0] else -1
            return float(0), o
        slope = cls.slope(p1, p2)
        if 0 < slope < 100000:
            return slope, 1 if p1[1] < p2[1] else -1
        if -100000 < slope < 0:
            return slope, 1 if p1[0] < p2[0] else -1

    @staticmethod
    def distance(pixel1, pixel2):
        return ((pixel2[1] - pixel1[1]) ** 2 + (pixel2[0] - pixel1[0]) ** 2) ** 0.5


class PILHandler(object):
    RADIUS = 5
    CIRCLE_COLOR = (0, 0, 255, 0)

    def __init__(self, data_source=None):
        self.image_path = data_source or 'ocr/images/example1.png'
        self.img = Image.open(self.image_path)
        self.pixels = None
        self._black_pixels = None
        self._two_dim_array = None

    @property
    def __size__(self):
        return self.img.size[0], self.img.size[1]

    @property
    def two_dim_array(self):
        if self._two_dim_array is None:
            self._two_dim_array = [
                (x, y) for y in range(self.__size__[1])
                for x in range(self.__size__[0])]
        return self._two_dim_array

    @property
    def black_pixels(self):
        if self._black_pixels is None:
            self.pixels = self.img.load()
            self._black_pixels = set(pixel for pixel in self.two_dim_array
                                     if Pixel.is_mostly_black(self.pixels[pixel[0], pixel[1]]))
        return self._black_pixels

    @classmethod
    def highlight_pixels(cls, img, pixels, color=None, no_circle=False):
        draw = ImageDraw.Draw(img)
        color = color or cls.CIRCLE_COLOR
        px = img.load()
        for node in pixels:
            if not no_circle:
                draw.ellipse((node[0] - cls.RADIUS,
                              node[1] - cls.RADIUS,
                              node[0] + cls.RADIUS,
                              node[1] + cls.RADIUS),
                              fill=color)
            else:
                px[node[0], node[1]] = Pixel.GREEN

        return img

    def create_img_from_pixel_map(self, pixel_list):
        img = Image.new('RGB', self.__size__, 'white')
        new_pixels = img.load()
        for pixel in pixel_list:
            x, y = pixel
            new_pixels[x, y] = Pixel.BLACK

        return img


class Tree(object):

    def __init__(self, item=None, parent=None, branches=None):
        if branches is None:
            branches = []

        self.item = item
        self.branches = branches
        self.parent = parent

    def __unicode__(self):
        return u"{}, branches: {}".format(str(len(self.item)), str(len(self.branches)))

    def __str__(self):
        return self.__unicode__()

    @property
    def length(self):
        return len(self.item)

    @property
    def is_leaf(self):
        return len(filter(None, self.branches)) == 0
