from django.db import models
from PIL import Image

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

    def __init__(self, image_path=None):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.pixels = self.img.load()
        self.max_density = (None, 0)
        self.identified_polygons = []
        self.instance_black_pixels = set()

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

        img.show()

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






