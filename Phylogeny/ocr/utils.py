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
        (UP, UP_LEFT, UP_RIGHT), #(UP, UP_RIGHT),
        (RIGHT, UP_RIGHT, DOWN_RIGHT),# (RIGHT, DOWN_RIGHT),
        (DOWN, DOWN_RIGHT, DOWN_LEFT),# (DOWN, DOWN_LEFT),
        (LEFT, DOWN_LEFT, UP_LEFT),# (LEFT, UP_LEFT),
        (UP, LEFT),
        (UP, RIGHT),
        (DOWN, LEFT),
        (DOWN, RIGHT)
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
    def highlight_pixels(cls, img, pixels, color=None):
        draw = ImageDraw.Draw(img)
        color = color or cls.CIRCLE_COLOR
        for node in pixels:
            draw.ellipse((node[0] - cls.RADIUS,
                          node[1] - cls.RADIUS,
                          node[0] + cls.RADIUS,
                          node[1] + cls.RADIUS),
                         fill=color)

        return img

    def create_img_from_pixel_map(self, pixel_list):
        img = Image.new('RGB', self.__size__, 'white')
        new_pixels = img.load()
        for pixel in pixel_list:
            x, y = pixel
            new_pixels[x, y] = Pixel.BLACK

        return img


class PolygonAnalyzer(object):

    def __init__(self, data_source=None):
        self.img = PILHandler(data_source)
        self.identified_polygons = []
        self.processed_tree = None
        self.instance_black_pixels = set()
        self._best_polygon = None
        self.contiguous_shape = set()

    @property
    def best_polygon(self):
        if self._best_polygon is None:
            self._best_polygon = max(self.identified_polygons, key=lambda x: x[1])[0]
        return self._best_polygon

    def execute_contiguous_walk(self, polygon):
        start = iter(polygon).next()
        self.contiguous_shape = set()
        self.contiguous_shape.add(start)
        checked_pixels = set()
        while True:
            new_pixels = set()
            for pixel in self.contiguous_shape:
                if pixel in checked_pixels:
                    continue
                else:
                    checked_pixels.add(pixel)

                for step in (Steps.step(pixel, step) for step in Steps.ALL):
                    if step in checked_pixels:
                        continue
                    if step in polygon:
                        new_pixels.add(step)

            if len(new_pixels):
                self.contiguous_shape |= new_pixels
            else:
                break

        return self.contiguous_shape

    def find_all_polygons(self):
        self.instance_black_pixels = set(self.img.black_pixels)
        while len(self.instance_black_pixels):
            polygon = self.execute_contiguous_walk(self.instance_black_pixels)
            self.instance_black_pixels = set(p for p in self.instance_black_pixels
                                             if p not in polygon)
            self.identified_polygons.append((polygon, len(polygon)))

    @staticmethod
    def edges(polygon, strict=False):
        new_pixels = set()
        for pixel in polygon:

            borders_white = False
            steps = Steps.ALL if not strict else Steps.MAIN_FOUR
            for test in (Steps.step(pixel, step) for step in steps):
                if test not in polygon:
                    borders_white = True
                    break

            if borders_white:
                new_pixels.add(pixel)

        return new_pixels

    def smart_reduction(self, polygon):
        thinner_polygon = polygon
        last_length = len(thinner_polygon)
        required_pixels = set()
        while True:
            edges = self.edges(thinner_polygon)
            print len(thinner_polygon), len(edges)

            for edge_pixel in set(edges):
                neighbors = Steps.step_all(edge_pixel, thinner_polygon)
                edge_neighbors = filter(lambda x: x in edges, neighbors)

                if len(neighbors) != len(edge_neighbors):
                    thinner_polygon.remove(edge_pixel)
                else:
                    required_pixels.add(edge_pixel)
            if len(thinner_polygon) == last_length:
                break
            else:
                last_length = len(thinner_polygon)

        return thinner_polygon

    def fix_gaps(self, thinner_polygon):
        floating_edges = set(n for p in thinner_polygon
                             for n in Steps.all_step(p)
                             if n not in thinner_polygon)

        candidates = set()
        for floating_edge in floating_edges:
            fneighbors = Steps.all_step(floating_edge)
            add_this = False

            for orient in (Steps.VERT, Steps.HORIZ):
                scored = [sum(1 if fneighbors[index] in thinner_polygon and index != -1 else 0
                              for index in row) for row in orient]

                top, mid, bottom = scored
                if mid == 0 and all(x > 0 for x in (top, bottom)):
                    add_this = True
                    break

            if add_this:
                candidates.add(floating_edge)

        print candidates
        new_set = set()
        neighbor_set = set()
        for candidate in candidates:
            neighboring_c = Steps.step_all(candidate, candidates)
            neighbor_set |= set(neighboring_c)
            if candidate not in neighbor_set:
                new_set.add(candidate)

        if len(new_set):
            print new_set
            thinner_polygon |= new_set

        return thinner_polygon

    def process_tree_polygon(self):
        if not len(self.identified_polygons):
            self.find_all_polygons()

        tree_structure = self.best_polygon
        thin_structure = self.smart_reduction(tree_structure)
        prc_structure = self.fix_gaps(thin_structure)

        self.processed_tree = prc_structure

        img = self.img.create_img_from_pixel_map(prc_structure)
        img.show()
        return prc_structure
