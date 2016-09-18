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
    UPPER_BOUND_FOR_BLACK = 100

    @classmethod
    def mostly_black(cls, pixel_values):
        for p in pixel_values:
            if p < cls.UPPER_BOUND_FOR_BLACK:
                return True
        return False

    @classmethod
    def mostly_white(cls, pixel_values):
        for p in pixel_values:
            if p > cls.LOWER_BOUND_FOR_WHITENESS:
                return True
        return False




class OpticalCharacterRecognition(object):

    def __init__(self, image_path=None):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.pixels = self.img.load()

    @property
    def __size__(self):
        return self.img.size[0], self.img.size[1]

    def __iter__(self):
        for x in range(self.img.size[0]):
            for y in range(self.img.size[1]):
                yield self.pixels[x, y]

    def yield_non_white_pixels(self):
        for pixel in self:
            for val in pixel:
                if val < self.LOWER_BOUND_FOR_WHITENESS:
                    yield pixel

    def yield_black_pixels(self):
        for pixel in self:
            for val in pixel:
                if val < self.UPPER_BOUND_FOR_BLACK:
                    yield pixel


    def create_new_image(self):
        img = Image.new('RGB', self.__size__, 'white')
        new_pixels = img.load()
        for i in range(self.__size__[0]):
            for j in range(self.__size__[1]):
                pixel = self.pixels[i, j]
                if len([val for val in pixel if val < self.UPPER_BOUND_FOR_BLACK]):
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
                if not Pixel.mostly_black(self.pixels[x, y]):
                    continue
                starting_pixel = (x, y)
                shape_pixels = self.execute_contiguous_walk(starting_pixel)
                if len(shape_pixels) > biggest_shape[1]:
                    biggest_shape = (shape_pixels, len(shape_pixels))

        return biggest_shape

    def execute_contiguous_walk(self, start):
        contiguous_shape_pixels = [start]
        checked_pixels = []
        for pixel in contiguous_shape_pixels:
            if pixel in checked_pixels:
                continue
            else:
                checked_pixels.append(pixel)

            for step in (Steps.step(pixel, step) for step in Steps.ALL):
                try:
                    if Pixel.mostly_black(self.pixels[step[0], step[1]]):
                        contiguous_shape_pixels.append(step)
                except IndexError:
                    continue

        return contiguous_shape_pixels






