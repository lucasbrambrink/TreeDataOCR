from django.db import models
from PIL import Image, ImageDraw
import pytesseract

class th(object):

    @staticmethod
    def add(t1, t2):
        return (t1[0] + t2[0],
                t1[1] + t2[1])

    @staticmethod
    def get_perimeter_of_2D(array):
        return [array[x][y] for x in range(len(array))
                for y in range(len(array[0])) if x
        ]


class TwoDim(object):

    @staticmethod
    def top(array):
        return array[0]

    @staticmethod
    def right(array):
        return [array[x][-1] for x in range(len(array))]

    @staticmethod
    def left(array):
        return [array[x][0] for x in range(len(array))]

    @staticmethod
    def bottom(array):
        return array[-1]

    @classmethod
    def all(cls, array):
        return cls.top(array), cls.right(array), cls.bottom(array), cls.left(array)


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


class CharOCR(object):

    @staticmethod
    def get_string(img):
        return pytesseract.image_to_string(img)


class Steps(object):
    UP = (0, -1)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (1, -1)
    DOWN = (0, 1)
    DOWN_LEFT = (-1, 1)
    DOWN_RIGHT = (1, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    ALL = (UP, UP_LEFT, UP_RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, LEFT, RIGHT)

    @staticmethod
    def step(pixel, step_type):
        return (pixel[0] + step_type[0],
                pixel[1] + step_type[1])


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


class OpticalCharacterRecognition(object):
    TWO_DIM_CACHE = '_cache_two_dim'
    DEFAULT_SQUARE_SIZE = 10
    DEFAULT_STEPS = 10
    ALLOWANCE_FACTOR = 2
    FUDGINESS = 50

    def __init__(self, image_path=None):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.pixels = self.img.load()
        self.max_density = (None, 0)
        self.identified_polygons = []
        self.instance_black_pixels = set()
        self.corners = []
        self.find_all_polygons()
        self.branching_sites = self.fancier_nodes()
        self.leaves = self.corner_nodes()

        self.char = CharOCR.get_string(Image.open(self.image_path))

    def is_black(self, pixel):
        try:
            return Pixel.is_mostly_black(self.pixel_values_by_address[pixel])
        except (IndexError, KeyError):
            return False

    def get_cache(self, key):
        key = '_{}'.format(key)
        if not hasattr(self, key):
            return None

        return getattr(self, key)

    def set_cache(self, key, value):
        key = '_{}'.format(key)
        setattr(self, key, value)
        return value

    @property
    def pixel_values_by_address(self):
        return self.get_cache('p') or self.set_cache('p', {
            pixel: self.pixels[pixel[0], pixel[1]]
            for pixel in self.two_dim_array
        })


    @property
    def __size__(self):
        return self.img.size[0], self.img.size[1]

    def find_max_density_area(self, pixel, size=None):
        square_limit = size or self.DEFAULT_SQUARE_SIZE
        black_pixels = []
        entire_set = set(self.black_pixels)
        for x in range(pixel[0], pixel[0] + square_limit):
            for y in range(pixel[1], pixel[1] + square_limit):
                test_pixel = (x, y)
                if test_pixel not in entire_set:
                    continue
                else:
                    black_pixels.append(test_pixel)

        if len(black_pixels) > self.max_density[1]:
            print black_pixels
            self.max_density = (pixel, len(black_pixels))

    @property
    def two_dim_array(self):
        return self.get_cache('t') or self.set_cache('t', value=[
                (x, y) for y in range(self.__size__[1])
                for x in range(self.__size__[0])
            ])

    @property
    def black_pixels(self):
        return self.get_cache('bp') or self.set_cache('bp', set([
            pixel for pixel in self.two_dim_array
            if Pixel.is_mostly_black(self.pixels[pixel[0], pixel[1]])
        ]))

    def __iter__(self):
        for x in range(self.img.size[0]):
            for y in range(self.img.size[1]):
                yield self.pixels[x, y]

    def create_new_image(self):
        img = Image.new('RGB', self.__size__, 'white')
        new_pixels = img.load()
        for i in range(self.__size__[0]):
            for j in range(self.__size__[1]):
                pixel = self.pixels[i, j]
                if len([val for val in pixel if val < Pixel.UPPER_BOUND_FOR_BLACK]):
                    new_pixels[i, j] = (0, 0, 0)

        img.show()

    def create_img_from_pixel_map(self, pixel_list):
        img = Image.new('RGB', self.__size__, 'white')
        new_pixels = img.load()
        for pixel in pixel_list:
            x, y = pixel
            new_pixels[x, y] = Pixel.BLACK

        return img

    def show_image(self, img):
        img.show()

    def show_pixel_map(self, pixel_list):
        return self.show_image(self.create_img_from_pixel_map(pixel_list))

    def highlight_nodes(self, img, nodes, color=None):
        draw = ImageDraw.Draw(img)
        RADIUS = 5
        CIRCLE_COLOR = color or (0, 0, 255, 0)
        for node in nodes:
            draw.ellipse((node[0] - RADIUS, node[1] - RADIUS, node[0] + RADIUS, node[1] + RADIUS), fill=CIRCLE_COLOR)

        return img

    def find_longest_contiguous_pixel_map(self):
        biggest_shape = (None, 0)
        for x in range(self.img.size[0]):
            for y in range(self.img.size[1]):
                if not Pixel.is_mostly_black(self.pixels[x, y]):
                    continue
                starting_pixel = (x, y)
                shape_pixels = self.execute_contiguous_walk(starting_pixel)
                if len(shape_pixels) > biggest_shape[1]:
                    biggest_shape = (shape_pixels, len(shape_pixels))

        return biggest_shape

    def find_all_polygons(self):
        self.instance_black_pixels = list(set(self.black_pixels))
        while len(self.instance_black_pixels):
            polygon_start = self.instance_black_pixels[0]
            polygon = self.execute_contiguous_walk(polygon_start)
            self.instance_black_pixels = [p for p in self.instance_black_pixels
                                          if p not in polygon]

            self.identified_polygons.append((polygon, len(polygon)))

    def show_largest_polygon(self):
        max_polygon = max(self.identified_polygons, key=lambda x: x[1])
        self.create_img_from_pixel_map(max_polygon[0])

    def edge_pixels(self, polygon, num_steps=None):
        num_steps = num_steps or self.DEFAULT_STEPS
        all_operations = [(x, y) for x in range(-num_steps, num_steps + 1)
                          for y in range(-num_steps, num_steps + 1)]

        black_pixels = set(self.black_pixels)
        least_white_number_pixels = (2 * (num_steps ** 2)) + num_steps - self.ALLOWANCE_FACTOR
        edge_pixels = []
        nodes = []
        for pixel in polygon:
            white_pixels = 0
            count_black_pixels = 0
            for operation in all_operations:
                test_pixel_position = th.add(pixel, operation)
                if test_pixel_position not in black_pixels:
                    white_pixels += 1
                else:
                    count_black_pixels += 1

            if white_pixels >= least_white_number_pixels:
                edge_pixels.append(pixel)

            nodes.append((pixel, count_black_pixels))

        return edge_pixels, nodes

    @property
    def best_polygon(self):
        return max(self.identified_polygons, key=lambda x: x[1])[0]

    def best_nodes(self):
        e, nodes = self.edge_pixels(self.best_polygon)
        best_third = len(nodes) / 3
        clean_nodes = [node[0] for node in sorted(nodes, key=lambda x: x[1], reverse=True)[:best_third]]
        best_nodes = [clean_nodes[0]]
        for node in clean_nodes:
            nothing_similar_exists = True
            for seen_node in best_nodes:
                if self.within_range(seen_node, node):
                    nothing_similar_exists = False
                    break

            if nothing_similar_exists:
                best_nodes.append(node)

        return best_nodes

    def show_nodes(self):
        # self.DEFAULT_STEPS = 20
        # self.FUDGINESS = 50
        # self.find_all_polygons()
        # self.branching_sites = self.fancier_nodes()
        # self.leaves = self.corner_nodes()
        best = [b[0] for b in self.branching_sites]
        lowest = [a[0] for a in self.leaves]

        img = self.create_img_from_pixel_map(self.best_polygon)
        img = self.highlight_nodes(img, best)
        img = self.highlight_nodes(img, lowest, Pixel.GREEN)

        img.show()

    def count_white_edges(self, pixel, edge_row_operations):
        white_edges = 0
        for row in edge_row_operations:
            if all(Steps.step(pixel[0], operation) not in self._bp
                   for operation in row):
                white_edges += 1

        return white_edges

    def get_edges(self, num_steps):
        return TwoDim.all([[(x, y) for x in range(-num_steps, num_steps + 1)] for y in range(-num_steps, num_steps + 1)])


    def corner_nodes(self):
        value = self.get_cache('cn')
        if value:
            return value


        leaves = []

        e, nodes = self.edge_pixels(self.best_polygon)
        cleaner_nodes = sorted(nodes, key=lambda x: x[1])[:len(nodes) / 2]
        best_leaves = self.filter_by_range(cleaner_nodes)
        step_size = 3
        for num_steps in range(3, self.DEFAULT_STEPS * 2):
            found_leaf_count = 0
            for leaf in best_leaves:
                white_edges = self.count_white_edges(leaf, self.get_edges(num_steps))
                if white_edges >= 3:
                    found_leaf_count += 1

            if found_leaf_count > (len(best_leaves) / 2):
                print "STOPPED AT", found_leaf_count
                step_size = num_steps
                break

        print "BEST STEP SIZE", step_size
        print "TEST LEAF", best_leaves

        edge_row_operations = self.get_edges(step_size)
        for pixel in cleaner_nodes:
            white_edges = self.count_white_edges(pixel, edge_row_operations)
            if white_edges >= 3:
                leaves.append(pixel)

        self.FUDGINESS = step_size * 1.5
        value = self.filter_by_range(leaves)

        return self.set_cache('cn', value)
        #
        #
        # clean_nodes = sorted(nodes, key=lambda x: x[1])
        # range_excluded = self.filter_by_range(clean_nodes)
        # first_std_filter = self.filter_by_std(range_excluded, lambda x: x < 0)
        # std = Stat([f[1] for f in first_std_filter]).std
        # value = self.filter_by_std(first_std_filter, lambda x: x <= std)
        # return self.set_cache('cn', value)

    def filter_by_range(self, clean_nodes):
        best_nodes = [clean_nodes[0]]
        for node in clean_nodes:
            nothing_similar_exists = True
            for seen_node in best_nodes:
                if self.within_range(seen_node[0], node[0]):
                    nothing_similar_exists = False
                    break

            if nothing_similar_exists:
                best_nodes.append(node)

        return best_nodes

    def filter_by_std(self, clean_nodes, comparison):
        stat = Stat([node[1] for node in clean_nodes])
        positive_diff = []
        for node, deviation in zip(clean_nodes, stat.deviations):
            if comparison(deviation):
                positive_diff.append(node)

        return positive_diff

    def fancier_nodes(self):
        value = self.get_cache('fn')
        if value:
            return value
        e, nodes = self.edge_pixels(self.best_polygon)
        best_third = len(nodes) / 3
        clean_nodes = [node for node in sorted(nodes, key=lambda x: x[1], reverse=True)[:best_third]]
        best_nodes = self.filter_by_range(clean_nodes)
        value = self.filter_by_std(best_nodes, lambda x: x > 0)
        return self.set_cache('fn', value)

    def within_range(self, node, second_node):
        distance = ((second_node[1] - node[1]) ** 2 + (second_node[0] - node[0]) ** 2) ** 0.5
        return distance < self.FUDGINESS

    def execute_contiguous_walk(self, start):
        contiguous_shape_pixels = set([start])
        checked_pixels = set()
        black_pixels = set(self.black_pixels)
        while True:
            new_pixels = set()
            for pixel in contiguous_shape_pixels:
                if pixel in checked_pixels:
                    continue
                else:
                    checked_pixels.add(pixel)

                for step in (Steps.step(pixel, step) for step in Steps.ALL):
                    if step in checked_pixels:
                        continue
                    if step in black_pixels:
                        new_pixels.add(step)

            if len(new_pixels):
                contiguous_shape_pixels |= new_pixels
            else:
                break

        return contiguous_shape_pixels


