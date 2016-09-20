from django.db import models
from PIL import Image, ImageDraw


class th(object):

    @staticmethod
    def add(t1, t2):
        return (t1[0] + t2[0],
                t1[1] + t2[1])

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
    ALL = (UP, DOWN, LEFT, RIGHT)

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
    DEFAULT_STEPS = 2
    ALLOWANCE_FACTOR = 2
    FUDGINESS = 5

    def __init__(self, image_path=None):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.pixels = self.img.load()
        self.max_density = (None, 0)
        self.identified_polygons = []
        self.instance_black_pixels = set()
        self.corners = []
        self.branching_sites = []
        self.leaves = []

    def is_black(self, pixel):
        try:
            return Pixel.is_mostly_black(self.pixel_values_by_address[pixel])
        except (IndexError, KeyError):
            return False

    @property
    def pixel_values_by_address(self):
        value = {pixel: self.pixels[pixel[0], pixel[1]]
                 for pixel in self.two_dim_array}

        return self.cached_property('pixel_values_by_address', value)

    @property
    def __size__(self):
        return self.img.size[0], self.img.size[1]

    def cached_property(self, property_name, value):
        key = '_{}'.format(property_name)
        if hasattr(self, key):
            return getattr(self, key)

        setattr(self, key, value)
        return value

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
        value = [(x, y) for y in range(self.__size__[1])
                 for x in range(self.__size__[0])]
        return self.cached_property('two_dim_array', value)

    @property
    def black_pixels(self):
        value = [pixel for pixel in self.two_dim_array
                 if Pixel.is_mostly_black(self.pixels[pixel[0], pixel[1]])]
        return self.cached_property('black_pixels', set(value))

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
            for pixel in polygon:
                self.instance_black_pixels.remove(pixel)

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
        return max(self.identified_polygons, key=lambda x: [1])[0]

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
        self.DEFAULT_STEPS = 20
        self.FUDGINESS = 50
        self.find_all_polygons()
        self.branching_sites = self.fancier_nodes()
        self.leaves = self.corner_nodes()
        best = [b[0] for b in self.branching_sites]
        lowest = [a[0] for a in self.leaves]

        img = self.create_img_from_pixel_map(self.best_polygon)
        img = self.highlight_nodes(img, best)
        img = self.highlight_nodes(img, lowest, Pixel.GREEN)

        img.show()

    def corner_nodes(self):
        e, nodes = self.edge_pixels(self.best_polygon)
        clean_nodes = sorted(nodes, key=lambda x: x[1])
        range_excluded = self.filter_by_range(clean_nodes)
        first_std_filter = self.filter_by_std(range_excluded, lambda x: x < 0)
        std = Stat([f[1] for f in first_std_filter]).std
        return self.filter_by_std(first_std_filter, lambda x: x <= std)


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
        e, nodes = self.edge_pixels(self.best_polygon)
        best_third = len(nodes) / 3
        clean_nodes = [node for node in sorted(nodes, key=lambda x: x[1], reverse=True)[:best_third]]
        best_nodes = self.filter_by_range(clean_nodes)
        return self.filter_by_std(best_nodes, lambda x: x > 0)

    def within_range(self, node, second_node):
        distance = ((second_node[1] - node[1]) ** 2 + (second_node[0] - node[0]) ** 2) ** 0.5
        return distance < self.FUDGINESS

    def execute_contiguous_walk(self, start):
        contiguous_shape_pixels = [start]
        checked_pixels = []
        for pixel in contiguous_shape_pixels:
            if pixel in checked_pixels:
                continue
            else:
                checked_pixels.append(pixel)

            for step in (Steps.step(pixel, step) for step in Steps.ALL):
                if step in checked_pixels:
                    continue
                try:
                    if Pixel.is_mostly_black(self.pixels[step[0], step[1]]):
                        contiguous_shape_pixels.append(step)
                except IndexError:
                    continue

        return set(contiguous_shape_pixels)






