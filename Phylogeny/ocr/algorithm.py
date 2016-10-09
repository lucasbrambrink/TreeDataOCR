from .utils import PILHandler, Steps


class PolygonAnalyzer(object):
    """
    Contains main algorithm
    1. Identify all separate contiguous black pixel maps (polygons)
    2. a visualization of a phylogenetic tree inherently will have the biggest polygon be the tree
    3. reduce the polygon of arbitrary thickness to a skelleton of ~= 1pixel thickness at any single point
    4. traverse tree starting at any given leaf, ignoring "noise" branches,
    ==> allows following unique path from any single point to another.
    thereby extract tree structure -- traverse like N-Tree (not nec. binary)
    """

    def __init__(self, data_source=None):
        self.img = PILHandler(data_source)
        self.identified_polygons = []
        self.processed_tree = None
        self.instance_black_pixels = set()
        self._best_polygon = None
        self.contiguous_shape = set()

    def process_tree_polygon(self):
        if not len(self.identified_polygons):
            self.find_all_polygons()

        # select tree polygon
        tree_structure = self.best_polygon

        # reduce all edges until all pixels in polygon are also edges
        thin_structure = self.smart_reduction(tree_structure)

        # fix any breaks in continuity
        amended_polygons = self.amend_polygons(thin_structure)

        # remove any duplicate / unnecessary pixels
        f = self.final_reduce(amended_polygons)

        self.processed_tree = f
        img = self.img.create_img_from_pixel_map(self.processed_tree)
        img.show()
        return self.processed_tree

    def amend_polygons(self, thin_structure):
        amended_polygons = set(thin_structure)
        last_length = len(thin_structure)
        while True:
            amended_polygons = self.fix_gaps_between_polygons(amended_polygons)
            if len(amended_polygons) == last_length:
                break
            else:
                last_length = len(amended_polygons)

        return amended_polygons

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

    def find_all_polygons(self, pixel_map=None):
        self.instance_black_pixels = pixel_map or set(self.img.black_pixels)
        polygons = []
        while len(self.instance_black_pixels):
            polygon = self.execute_contiguous_walk(self.instance_black_pixels)
            self.instance_black_pixels = set(p for p in self.instance_black_pixels
                                             if p not in polygon)
            polygons.append((polygon, len(polygon)))

        if pixel_map is None:
            self.identified_polygons = polygons

        return polygons

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
        while True:
            edges = self.edges(thinner_polygon)
            for edge_pixel in set(edges):
                neighbors = Steps.step_all(edge_pixel, thinner_polygon)
                edge_neighbors = filter(lambda x: x in edges, neighbors)

                if len(neighbors) != len(edge_neighbors):
                    thinner_polygon.remove(edge_pixel)

            if len(thinner_polygon) == last_length:
                break
            else:
                last_length = len(thinner_polygon)

        return thinner_polygon

    def fix_gaps_between_polygons(self, thinner_polygon):
        polygons = self.find_all_polygons(thinner_polygon)
        new_set = set()
        print len(polygons)
        for polygon in polygons:
            fixed = self.fix_gaps(polygon[0], thinner_polygon)
            new_set |= fixed

        return new_set

    def fix_gaps(self, polygon, full_map):
        floating_edges = set(n for p in polygon
                             for n in Steps.all_step(p)
                             if n not in polygon)

        candidates = set()
        for floating_edge in floating_edges:
            fneighbors = Steps.step_many(floating_edge, 2)
            add_this = False

            for orient in (Steps.VERT, Steps.HORIZ):
                scored = [sum(1 if (fneighbors[index] in full_map and
                                    fneighbors[index] not in polygon and
                                    index != -1) else 0
                              for index in row) for row in orient]

                if any(x > 0 for x in scored):
                    add_this = True
                    break

            if add_this:
                candidates.add(floating_edge)

        new_set = set()
        neighbor_set = set()
        for candidate in candidates:
            neighboring_c = Steps.step_all(candidate, candidates)
            neighbor_set |= set(neighboring_c)
            if candidate not in neighbor_set:
                new_set.add(candidate)

        if len(new_set):
            polygon |= new_set

        return polygon

    @staticmethod
    def final_reduce(thinner_polygon):
        last_length = len(thinner_polygon)
        checked = set()
        while True:
            for floating_edge in thinner_polygon:
                if floating_edge in checked:
                    continue
                else:
                    checked.add(floating_edge)

                fneighbors = Steps.all_step(floating_edge)
                save_to_remove = False

                for orient in (Steps.VERT, Steps.HORIZ):
                    scored_rows = [tuple(1 if fneighbors[index] in thinner_polygon and index != -1 else 0
                                   for index in row) for row in orient]
                    scored = [sum(row) for row in scored_rows]

                    top, mid, bottom = scored
                    if (top == 3 and bottom == 0) or (bottom == 3 and top == 0):
                        save_to_remove = True
                        break

                if save_to_remove:
                    thinner_polygon.remove(floating_edge)
                    break

            if len(thinner_polygon) == last_length:
                break
            else:
                last_length = len(thinner_polygon)

        return thinner_polygon
