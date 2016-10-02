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
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (1, -1)
    DOWN_LEFT = (-1, 1)
    DOWN_RIGHT = (1, 1)
    ALL = (UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT)
    MAIN_FOUR = (UP, DOWN, LEFT, RIGHT)

    QUADRANTS = (
        (UP, RIGHT, UP_RIGHT),
        (DOWN, RIGHT, DOWN_RIGHT),
        (DOWN, LEFT, DOWN_LEFT),
        (UP, LEFT, UP_LEFT),
    )

    ORIENTATIONS = (
        (QUADRANTS[0], QUADRANTS[2]),
        (QUADRANTS[1], QUADRANTS[3]),
        (QUADRANTS[2], QUADRANTS[0]),
        (QUADRANTS[3], QUADRANTS[1])
    )

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

    @staticmethod
    def slope(pixel1, pixel2):
        if (pixel2[0] - pixel1[0]) == 0:
            return float(100)
        return float(pixel2[1] - pixel1[1]) / float(pixel2[0] - pixel1[0])

    @staticmethod
    def distance(pixel1, pixel2):
        return ((pixel2[1] - pixel1[1]) ** 2 + (pixel2[0] - pixel1[0]) ** 2) ** 0.5


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
        # self.branching_sites = self.fancier_nodes()
        # self.leaves = self.corner_nodes()

        # self.char = CharOCR.get_string(Image.open(self.image_path))

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

    # def find_line_thickness(self, polygon):
    #     if len(polygon) == 0:
    #         raise Exception("Polygon is empty")
    #
    #     radius = 1
    #     all_operations = [(x, y) for x in range(-radius, radius + 1)
    #                       for y in range(-radius, radius + 1)]
    #     best_pixel = (polygon[0], 0)
    #     black_pixels = set(self.black_pixels)
    #     current_pixel = best_pixel
    #     count = 0
    #     for operation in all_operations:
    #         test_pixel_position = th.add(pixel, operation)
    #         if Pixel.distance(test_pixel_position, current_pixel) > radius:
    #             continue
    #
    #         if test_pixel_position in black_pixels:
    #             count += 1
    #
    #         if count > best_pixel[1]:
    #             best_pixel = (current_pixel, count)
    #
    #
    #
    #
    #
    #             if test_pixel_position not in black_pixels:
    #                 white_pixels += 1


    #
    # def perform_tree_analysis(self, polygon_map):
    #     # starting_pixel = sorted(polygon_map, key=lambda x: (x[0], x[1]))[0]
    #     # seen_pixels = set()
    #
    #     num_steps = num_steps or self.DEFAULT_STEPS
    #
    #     least_white_number_pixels = (2 * (num_steps ** 2)) + num_steps - self.ALLOWANCE_FACTOR
    #     edge_pixels = []
    #     nodes = []
    #     for pixel in polygon_map:
    #         white_pixels = 0
    #         count_black_pixels = 0
    #         for operation in all_operations:
    #             test_pixel_position = th.add(pixel, operation)
    #             if test_pixel_position not in black_pixels:
    #                 white_pixels += 1
    #             else:
    #                 count_black_pixels += 1
    #
    #         if white_pixels >= least_white_number_pixels:
    #             edge_pixels.append(pixel)




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
















class Skelleton(object):
    """
    there is a point at which
    the set of all the best pixels (center)
    will remain constant from n to n + 1



    """

    def __init__(self):
        pass

    @staticmethod
    def min_score(scored_pixels):
        return min(scored_pixels, key=lambda x: x[1])[1]

    @staticmethod
    def nth_lowest(array, n):
        return sorted(array, key=lambda x: x[1])[n - 1]

    @classmethod
    def count_mins(cls, scored_pixels):
        minimum = cls.min_score(scored_pixels)
        return sum(1 for p in scored_pixels if p[1] == minimum)

    @classmethod
    def count_below_than(cls, n, scored_pixels):
        nth_lowest = cls.nth_lowest(scored_pixels, n)
        return sum(1 for p in scored_pixels if p[1] <= nth_lowest)

    @staticmethod
    def score_pixels(steps, polygon):
        operations = [(step_x, step_y) for step_x in range(-steps, steps + 1)
                      for step_y in range(-steps, steps + 1)
                      if Pixel.distance((0,0), (step_x, step_y)) <= steps]
        scored_pixels = []
        for pixel in polygon:
            score = sum(1 for other_pixel in
                        (Steps.step(pixel, step) for step in operations)
                        if other_pixel in polygon)
            scored_pixels.append((pixel, score))
        return scored_pixels

    @staticmethod
    def edges(polygon):
        new_pixels = set()
        for pixel in polygon:

            borders_white = False
            for step in (Steps.step(pixel, step) for step in Steps.ALL):
                if step not in polygon:
                    borders_white = True
                    break

            if borders_white:
                new_pixels.add(pixel)

        return new_pixels

    @staticmethod
    def strict_edges(polygon):
        new_pixels = set()
        for pixel in polygon:

            borders_white = False
            for step in (Steps.step(pixel, step)
                         for step in Steps.MAIN_FOUR):
                if step not in polygon:
                    borders_white = True
                    break

            if borders_white:
                new_pixels.add(pixel)

        return new_pixels

    @staticmethod
    def order_polygon(polygon):
        seen_pixels = set()
        current_pixel = iter(polygon).next()
        contiguous_ordering = [current_pixel]
        while True:
            found_new_pixels = []
            for step in Steps.ALL:
                test = Steps.step(current_pixel, step)
                if test in seen_pixels:
                    continue

                if test in polygon:
                    found_new_pixels.append(test)
                    seen_pixels.add(test)

            if len(found_new_pixels) == 1:
                contiguous_ordering.append(found_new_pixels[0])
                current_pixel = found_new_pixels[0]
            if len(found_new_pixels) > 1:
                possible = []
                for found_pixel in found_new_pixels:
                    can_continue = False
                    for step in Steps.ALL:
                        test_found = Steps.step(found_pixel, step)
                        if test_found in seen_pixels:
                            continue
                        if test_found in polygon:
                            can_continue = True
                            break

                    if can_continue:
                        possible.append(found_pixel)

                if len(possible) > 0:
                    current_pixel = possible[0]
                    contiguous_ordering.append(current_pixel)
                else:
                    found_new_pixels = []

            if len(contiguous_ordering) == len(polygon):
                break

            if not len(found_new_pixels):
                last_added = contiguous_ordering[-1]
                for step in Steps.ALL:
                    test = Steps.step(current_pixel, step)
                    if test in polygon and test != last_added:
                        found_new_pixels.append(test)

                for finishing_pixel in found_new_pixels:
                    contiguous_ordering.append(finishing_pixel)

                break

        return contiguous_ordering



    @staticmethod
    def find_n_neighbors(pixel, available_steps, n, polygon):
        found = []
        current_pixel = pixel
        while True:
            found_new_pixel = False
            for step in available_steps:
                test = Steps.step(current_pixel, step)
                if test in polygon:
                    found.append(test)
                    current_pixel = test
                    found_new_pixel = True
                    break

            if len(found) == n:
                return found
            elif not found_new_pixel:
                break

        return None



    @classmethod
    def surrounding_pixels(cls, pixel, polygon):
        RANGE = 4
        for orientation in Steps.ORIENTATIONS:
            seen = set()

            found = [cls.find_n_neighbors(pixel, direction, RANGE, polygon)
                     for direction in orientation]
            if all(found):
                return found

        return None

    @classmethod
    def score_polygon(cls, polygon):
        scored_pixels = []
        for pixel in polygon:
            surrounding = cls.surrounding_pixels(pixel, polygon)
            if surrounding is None:
                continue

            x, y = surrounding[0][-1], surrounding[1][-1]
            scored_pixels.append((pixel, Pixel.slope(x, y)))
            print x, y
            print surrounding
            print scored_pixels
            break
        return scored_pixels


    @staticmethod
    def score_pixel(index, ordered_polygon):
        STEPS = 3
        below = index - STEPS
        upper = index + STEPS if index + STEPS < len(ordered_polygon) else (index + STEPS) - len(ordered_polygon)
        diff = Pixel.slope(ordered_polygon[below], ordered_polygon[upper])
        return diff

    @classmethod
    def score_all_pixels(cls, polygon):
        return [(pixel, cls.score_pixel(index, polygon))
                for index, pixel in enumerate(polygon)]

    @staticmethod
    def compare_slopes(index, ordered_polygon):
        STEPS = 5
        if len(ordered_polygon) < STEPS:
            raise Exception(u"This polygon is almost empty")

        below = index - STEPS
        upper = index + STEPS + 1
        # print 'upper', upper, 'below', below
        beneath = []
        above = []
        for i in range(below, upper):
            if i >= len(ordered_polygon):
                i -= len(ordered_polygon)
                above.append(ordered_polygon[i][1])
                continue

            pixel = ordered_polygon[i][1]
            if i == index:
                continue
            elif i > index:
                above.append(pixel)
            else:
                beneath.append(pixel)

        return beneath[0] - above[-1]

    @classmethod
    def assign_secondary_scores(cls, scored):
        return [(scored_p[0], scored_p[1], cls.compare_slopes(index, scored))
                for index, scored_p in enumerate(scored)]

    def leaves(self, polygon):
        if type(polygon) is not set:
            polygon = set(polygon)

        steps = 5
        leaves = None

        reduced_polygon = set()
        first_item = iter(polygon).next()
        reduced_polygon.add(first_item)
        for pixel in polygon:
            if pixel in reduced_polygon:
                continue

            add = True
            for rp in reduced_polygon:
                if Pixel.distance(rp, pixel) < 5:
                    add = False
                    break

            if add:
                reduced_polygon.add(pixel)

        while True:
            scored_pixels = self.score_pixels(steps, reduced_polygon)
            plus_one_pixels = self.score_pixels(steps + 1, reduced_polygon)

            count_scored = self.count_below_than(2, scored_pixels)
            print count_scored
            print "Step:", steps
            # print scored_pixels
            if count_scored > 5 and count_scored == self.count_below_than(2, plus_one_pixels):
                minimum = self.min_score(scored_pixels)
                leaves = [pixel for pixel in scored_pixels if pixel[1] == minimum]
                break
            else:
                steps += 1

            if steps == 20:
                break

        return leaves


    @staticmethod
    def score(pixel, radius):
        pass

    @staticmethod
    def reduce_polygon_to_single_pixel_line(polygon):
        """
        take polygon
        identify leaves --> hard?
            -


        remove pixels that are not leaves
            - if still continuous (all leaves contained in a contiguous walk)
                remove
            else: dont remove

        once a state is reached where no pixel can be removed
        ==> resulting 2d polygon of thickness 1pixel --> trivial to find connections
        """



        pass

    @staticmethod
    def execute_contiguous_walk(polygon, starting_pixel):
        contiguous_shape_pixels = set()
        contiguous_shape_pixels.add(starting_pixel)

        checked_pixels = set()
        if type(polygon) is not set:
            polygon = set(polygon)

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
                    if step in polygon:
                        new_pixels.add(step)

            if len(new_pixels):
                contiguous_shape_pixels |= new_pixels
            else:
                break

        return contiguous_shape_pixels


    @classmethod
    def distill_leaves(cls, scored_pixels):
        numbers = [x[2] for x in scored_pixels]
        s = Stat(numbers)
        std = s.std

        outliers = [scored_pixel for scored_pixel, deviation
                    in zip(scored_pixels, s.deviations) if abs(deviation) > std]
        leaves = []
        for test in outliers:
            add_leaf = True
            distances = [(test[0], Pixel.distance(test[0], out[0]))
                         for out in outliers
                         if out[0] != test[0]]
            std = Stat([d[1] for d in distances]).std
            for leaf in leaves:
                if Pixel.distance(test[0], leaf[0]) < std * 0.5:
                    add_leaf = False
                    break

            if add_leaf:
                leaves.append(test)

        return leaves

    @classmethod
    def valid_state(cls, polygon, leaves):
        start = iter(polygon).next()
        contiguous = cls.execute_contiguous_walk(polygon, start)
        valid = True
        for leaf in leaves:
            if leaf not in contiguous:
                valid = False
                break

        return valid

    @classmethod
    def distill_polygon(cls, polygon, leaves):
        leaves = set([l[0] for l in leaves])
        new_polygon = polygon
        last_length = len(new_polygon)
        while True:
            for pixel in new_polygon:
                test_p = set([p for p in new_polygon
                              if p != pixel])
                if cls.valid_state(test_p, leaves):
                    new_polygon = test_p
                    break

            if len(new_polygon) == last_length:
                break
            else:
                last_length = len(new_polygon)

        return new_polygon



    @classmethod
    def test(cls):
        o = OpticalCharacterRecognition('ocr/images/example1.png')
        e = cls.strict_edges(o.best_polygon)
        op = cls.order_polygon(e)
        sp = cls.score_all_pixels(op)
        ssp = cls.assign_secondary_scores(sp)

        leaves = cls.distill_leaves(ssp)
        new_p = cls.distill_polygon(o.best_polygon, leaves)
        i = o.create_img_from_pixel_map(new_p)

        # leaf_map = [l[0] for l in leaves]
        # i = o.create_img_from_pixel_map(o.best_polygon)
        # i = o.highlight_nodes(i, leaf_map)
        i.show()

        data = {
            'image': o,
            'edges': e,
            'ordered_polygon': op,
            'scored': ssp,
            'outliers': outliers,
            'stat': s,
            'leaves': leaves,
            'lm': leaf_map

        }
        return data












